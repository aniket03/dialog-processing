import copy
import collections
import math
import random
import json
import code

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataSourceUnsupervised():

    def __init__(self, data, config, tokenizer):
        # Attributes
        # Attributes from config
        self.dataset_path = config.dataset_path
        self.max_uttr_len = config.max_uttr_len
        self.history_len = config.history_len
        # Other attributes
        self.tokenizer = tokenizer
        self.statistics = {"n_sessions": 0, "n_uttrs": 0, "n_tokens": 0, "n_segments": 0}

        sessions = data

        # Process sessions
        for sess in sessions:
            for uttr in sess["utterances"]:
                text = uttr["text"]
                floor = uttr["floor"]
                tokens = self.tokenizer.convert_string_to_tokens(text)[:self.max_uttr_len]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens, bos_and_eos=True)
                floor_id = ["A", "B"].index(floor)

                uttr.update({
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "floor_id": floor_id,
                    "is_next": True,
                })

        # Construct utterance pool for negative sampling
        self.utterance_pool = []
        for sess in sessions:
            for uttr in sess["utterances"]:
                self.utterance_pool.append(uttr)

        # Get segments
        self._segments = []
        for sess in sessions:
            uttrs = sess["utterances"]
            for segment_end_idx in range(1, len(uttrs)):

                # positive sample
                segment_start_idx = max(0, segment_end_idx-self.history_len)
                segment = {
                    "utterances": uttrs[segment_start_idx:segment_end_idx+1],
                    "segment_meta": {}
                }
                positive_uttr = uttrs[segment_end_idx]
                positive_uttr["utterance_meta"]["reference_uttr"] = positive_uttr
                self._segments.append(segment)

                # negative sample
                positive_uttr_token_ids = positive_uttr["token_ids"]
                while True:
                    negative_uttr = random.choice(self.utterance_pool)
                    negative_uttr_token_ids = negative_uttr["token_ids"]
                    if self._are_different_uttrs(positive_uttr_token_ids, negative_uttr_token_ids):
                        break
                negative_uttr = copy.deepcopy(negative_uttr)
                negative_uttr["utterance_meta"]["reference_uttr"] = positive_uttr
                negative_uttr["is_next"] = False
                segment = {
                    "utterances": uttrs[segment_start_idx:segment_end_idx] + [negative_uttr],
                    "segment_meta": {}
                }
                self._segments.append(segment)

        # Calculate basic statistics
        self.statistics["n_sessions"] = len(sessions)
        self.statistics["n_segments"] = len(self._segments)
        for sess in sessions:
            self.statistics["n_uttrs"] += len(sess["utterances"])
            for uttr in sess["utterances"]:
                tokens = uttr["text"].split(" ")
                self.statistics["n_tokens"] += len(tokens)

    def _are_different_uttrs(self, ids1, ids2):
        id_set1 = set(ids1)
        id_set2 = set(ids2)
        intersection = id_set1.intersection(id_set2)
        coverage_score = (len(intersection)/len(id_set1) + len(intersection)/len(id_set2))/2
        if coverage_score < 0.8:
            return True
        else:
            return False

    def epoch_init(self, shuffle=False):
        self.cur_segment_idx = 0
        if shuffle:
            self.segments = copy.deepcopy(self._segments)
            random.shuffle(self.segments)
        else:
            self.segments = self._segments

    def __len__(self):
        return len(self._segments)

    def next(self, batch_size, return_paired_Y=False):
        # Return None when running out of segments
        if self.cur_segment_idx == len(self.segments):
            return None

        # Data to fill in
        X, Y, Y_ref = [], [], []
        X_floor, Y_floor = [], []
        Y_is_next = []

        empty_sent = ""
        empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens, bos_and_eos=True)
        padding_uttr = {
            "tokens": empty_tokens,
            "token_ids": empty_ids,
            "floor_id": 0,
        }

        while self.cur_segment_idx < len(self.segments):
            if len(Y) == batch_size:
                break

            segment = self.segments[self.cur_segment_idx]
            segment_uttrs = segment["utterances"]
            self.cur_segment_idx += 1

            # NOTE: to return paired (Y, Y_ref), all Ys should be negative-sampleing responses
            if return_paired_Y and segment_uttrs[-1]["is_next"]:
                continue

            # First non-padding input uttrs
            for uttr in segment_uttrs[:-1]:
                X.append(uttr["token_ids"])
                X_floor.append(uttr["floor_id"])

            # Then padding input uttrs
            for _ in range(self.history_len-len(segment_uttrs)+1):
                uttr = padding_uttr
                X.append(uttr["token_ids"])
                X_floor.append(uttr["floor_id"])

            # Last output uttr
            uttr = segment_uttrs[-1]
            Y.append(uttr["token_ids"])
            Y_floor.append(uttr["floor_id"])
            Y_is_next.append(uttr["is_next"])

            # reference output uttr
            ref_uttr = segment_uttrs[-1]["utterance_meta"]["reference_uttr"]
            Y_ref.append(ref_uttr["token_ids"])

        X = self.tokenizer.convert_batch_ids_to_tensor(X)
        Y = self.tokenizer.convert_batch_ids_to_tensor(Y)
        Y_ref = self.tokenizer.convert_batch_ids_to_tensor(Y_ref)

        batch_size = Y.size(0)
        history_len = X.size(0)//batch_size

        X = torch.LongTensor(X).to(DEVICE).view(batch_size, history_len, -1)
        X_floor = torch.LongTensor(X_floor).to(DEVICE).view(batch_size, history_len)

        Y = torch.LongTensor(Y).to(DEVICE)
        Y_ref = torch.LongTensor(Y_ref).to(DEVICE)
        Y_floor = torch.LongTensor(Y_floor).to(DEVICE)
        Y_is_next = torch.BoolTensor(Y_is_next).to(DEVICE)

        batch_data_dict = {
            "X": X,
            "X_floor": X_floor,
            "Y": Y,
            "Y_ref": Y_ref,
            "Y_floor": Y_floor,
            "Y_is_next": Y_is_next,
        }

        return batch_data_dict
