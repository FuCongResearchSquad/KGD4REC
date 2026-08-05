# -*- coding: UTF-8 -*-

import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad

from helpers.BaseReader import BaseReader
from utils import utils
from utils.constants import *


class PretrainReader(BaseReader):

    @staticmethod
    def parse_data_args(parser):
        parser.add_argument(
            "--path",
            type=str,
            default="processed_llo_graph",
            help="Input data dir.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="Software",
            help="Choose a dataset.",
        )
        parser.add_argument(
            "--sep",
            type=str,
            default=",",
            help="sep of csv file.",
        )
        parser.add_argument(
            "--semantic_threshold",
            type=float,
            default=0.8,
            help="Cosine-similarity threshold for text embeddings.",
        )
        parser.add_argument(
            "--collaborative_threshold",
            type=float,
            default=0.5,
            help="Cosine-similarity threshold for graph embeddings.",
        )
        parser.add_argument(
            "--text_emb_path",
            type=str,
            default="",
            help="Optional path to pre-trained text embedding.csv. "
                 "If empty, defaults to text_emb/{dataset}/text_emb.csv",
        )
        parser.add_argument(
            "--graph_emb_path",
            type=str,
            default="",
            help="Optional path to pre-trained graph embedding.csv. "
                 "If empty, defaults to graph_emb/{dataset}/graph_emb.csv",
        )
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self.threshold_value = {
            "text": args.semantic_threshold,
            "graph": args.collaborative_threshold,
        }
        self.emb_path = {
            "text": args.text_emb_path,
            "graph": args.graph_emb_path,
        }
        self._threshold_item_emb_cache = {}
        logging.info(
            "DNTP config: text_thr=%.4f, graph_thr=%.4f  "
            "(NOTE: use --regenerate 1 when changing threshold settings to avoid stale cache)",
            self.threshold_value["text"],
            self.threshold_value["graph"],
        )
        self._read_data()

    def _resolve_emb_path(self, emb_path: str, emb_type: str):
        if emb_path == "":
            emb_path = os.path.join(f"{emb_type}_emb/{self.dataset}/{emb_type}_emb.csv")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")
        return emb_path

    def _parse_emb_string(self, s):
        """Parse a serialized vector string like '[0.1, 0.2, ...]' into a 1-D numpy array."""
        return np.fromstring(s.strip().strip("[]"), sep=",", dtype=np.float32)

    def _load_threshold_item_emb(self, emb_type: str):
        """Load external item embeddings and remap them to internal item ids."""
        if emb_type in self._threshold_item_emb_cache:
            return self._threshold_item_emb_cache[emb_type]

        emb_path = self._resolve_emb_path(self.emb_path[emb_type], emb_type)
        logging.info("Loading %s threshold embedding from %s", emb_type, emb_path)
        emb_df = pd.read_csv(emb_path, sep=self.sep)

        id_col = emb_df.columns[0]
        value_cols = list(emb_df.columns[1:])

        if len(value_cols) == 1:
            vectors = emb_df[value_cols[0]].apply(self._parse_emb_string)
            emb_matrix = np.stack(vectors.values)
        else:
            emb_matrix = emb_df[value_cols].values.astype(np.float32)

        emb_dim = emb_matrix.shape[1]
        item_emb = torch.zeros(self.n_items, emb_dim, dtype=torch.float32)
        for i, orig_id in enumerate(emb_df[id_col].values):
            item_id = self.orig_item_id2item_id.get(orig_id)
            if item_id is not None and 0 < item_id < self.n_items:
                item_emb[item_id] = torch.from_numpy(emb_matrix[i])

        norms = item_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        item_emb = item_emb / norms
        item_emb[0] = 0.0

        self._threshold_item_emb_cache[emb_type] = item_emb
        logging.info("Threshold embedding [%s] loaded: shape=%s", emb_type, tuple(item_emb.shape))
        return item_emb

    def _create_threshold_mask(self, item_seq_ids, emb_type: str):
        """Compute [N, L, L] masks using text or graph embedding cosine similarity."""
        if emb_type not in ("text", "graph"):
            raise ValueError(f"Unknown embedding type for DNTP labels: {emb_type}")

        item_emb = self._load_threshold_item_emb(emb_type)       # [n_items, d]
        seq_emb = item_emb[item_seq_ids]                         # [N, L, d]
        sim = torch.matmul(seq_emb, seq_emb.transpose(1, 2))     # [N, L, L]
        mask = sim > self.threshold_value[emb_type]              # [N, L, L]

        pad_mask = item_seq_ids == 0
        mask = mask & ~pad_mask.unsqueeze(2) & ~pad_mask.unsqueeze(1)
        return mask

    def _create_ntp_labels(self, item_seq_ids, item_seq_lens, label_method: str, emb_type=None):
        n_rows, seq_len = item_seq_ids.shape
        device = item_seq_ids.device

        pos = torch.arange(seq_len, device=device)
        future_pos_mask = (pos.unsqueeze(0) > pos.unsqueeze(1)).unsqueeze(0)

        if label_method == "next":
            threshold_mask = torch.ones(
                n_rows,
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            )
        elif label_method == "threshold":
            threshold_mask = self._create_threshold_mask(item_seq_ids, emb_type)
        else:
            raise ValueError(f"Unknown label method: {label_method}")

        label_mask = future_pos_mask & threshold_mask
        any_valid = label_mask.any(dim=-1)
        label_positions = label_mask.long().argmax(dim=-1)

        labels = torch.gather(item_seq_ids, 1, label_positions)
        labels[~any_valid] = -100

        pad_positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pad_lens = (seq_len - item_seq_lens).unsqueeze(1)
        labels[pad_positions < pad_lens] = -100
        return labels

    def _read_data(self):
        if self.dataset in [
            "CDs_and_Vinyl",
            "Software",
            "Video_Games",
            "Beauty_and_Personal_Care",
            "Office_Products",
            "Arts_Crafts_and_Sewing",
            "Cell_Phones_and_Accessories",
            "Toys_and_Games",
        ]:
            orig_item_id = "parent_asin"
            item_data_column = ["parent_asin", "item_id", "text_emb"]
            item_feat_column = []
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        self.item_feat_column = item_feat_column

        logging.info(f'Reading data from "{self.prefix}", dataset = "{self.dataset}"')

        item_df = pd.read_csv(
            os.path.join(self.prefix, self.dataset, f"{self.dataset}.item.csv"),
            sep=self.sep,
            usecols=item_data_column,
        ).reset_index(drop=True)

        orig_item_id2item_id = item_df.set_index(orig_item_id)[ITEM_ID].to_dict()
        self.orig_item_id2item_id = orig_item_id2item_id
        item_df.drop(columns=[orig_item_id], inplace=True)

        self.item_id2text_emb = item_df.set_index(ITEM_ID)["text_emb"].to_dict()
        item_df.drop(columns=["text_emb"], inplace=True)

        for feat in item_feat_column:
            if feat.endswith("seq_id"):
                item_df[feat] = item_df[feat].apply(eval)
        self.item_id2feat = item_df.set_index(ITEM_ID).to_dict(orient="index")
        self.item_feat_num = {
            feat: len(set(item_df[feat].dropna().explode())) for feat in item_feat_column
        }
        logging.info(f"item_feat_num: {self.item_feat_num}")

        inter_df = dict()
        user_set = set()
        n_entry = 0
        for key in ["train", "valid", "test"]:
            split_df = pd.read_csv(
                os.path.join(self.prefix, self.dataset, f"{self.dataset}.{key}.csv"),
                sep=self.sep,
            ).reset_index(drop=True)
            split_df = utils.eval_list_columns(split_df)
            split_df[ITEM_SEQ] = split_df[ITEM_SEQ].str.split()
            split_df[ITEM_SEQ_LEN] = split_df[ITEM_SEQ].apply(len)
            split_df[ITEM_ID] = split_df[orig_item_id].map(orig_item_id2item_id)
            split_df[ITEM_SEQ_ID] = split_df[ITEM_SEQ].apply(
                lambda x: [orig_item_id2item_id[iid] for iid in x]
            )
            user_set.update(split_df[USER_ID].tolist())
            n_entry += len(split_df)
            inter_df[key] = split_df

        logging.info("Counting dataset statistics...")
        self.n_users = len(user_set) + 1
        self.n_items = item_df[ITEM_ID].max() + 1
        del user_set
        logging.info(
            f'"# user": {self.n_users - 1}, "# item": {self.n_items - 1}, "# entry": {n_entry}'
        )

        data_dict = {key: dict() for key in ["train", "valid", "test"]}

        train_df = inter_df["train"]
        idx = train_df.groupby(USER_ID)[ITEM_SEQ_LEN].idxmax()
        train_longest = train_df.loc[idx].copy()
        train_longest["full_seq"] = train_longest.apply(
            lambda r: r[ITEM_SEQ_ID] + [int(r[ITEM_ID])],
            axis=1,
        )
        train_longest["full_seq"] = train_longest["full_seq"].apply(
            lambda x: x[-(MAX_ITEM_SEQ_LEN + 1):]
        )
        train_longest[ITEM_SEQ_LEN] = train_longest["full_seq"].apply(len)

        train_item_seq = [
            torch.from_numpy(np.array(x)).long() for x in train_longest["full_seq"].values
        ]
        train_left_padded = [
            pad(seq, (MAX_ITEM_SEQ_LEN + 1 - len(seq), 0), value=0) for seq in train_item_seq
        ]
        train_seq_ids = torch.stack(train_left_padded)
        train_seq_lens = torch.from_numpy(train_longest[ITEM_SEQ_LEN].values).long()

        item_labels_full = self._create_ntp_labels(
            train_seq_ids,
            train_seq_lens,
            "next",
        )
        text_labels_full = self._create_ntp_labels(
            train_seq_ids,
            train_seq_lens,
            "threshold",
            "text",
        )
        graph_labels_full = self._create_ntp_labels(
            train_seq_ids,
            train_seq_lens,
            "threshold",
            "graph",
        )

        train_seq_ids = train_seq_ids[:, :-1]
        item_labels = item_labels_full[:, :-1]
        text_labels = text_labels_full[:, :-1]
        graph_labels = graph_labels_full[:, :-1]
        train_seq_lens = train_seq_lens - 1

        valid_mask = (
            (item_labels != -100).any(dim=1)
            | (text_labels != -100).any(dim=1)
            | (graph_labels != -100).any(dim=1)
        )
        dropped = (~valid_mask).sum().item()
        logging.info(
            "DNTP: drop %d training samples with no valid labels across "
            "all three loss types, now %d left.",
            dropped,
            valid_mask.sum().item(),
        )

        data_dict["train"][ITEM_SEQ_ID] = train_seq_ids[valid_mask]
        data_dict["train"][ITEM_SEQ_LEN] = train_seq_lens[valid_mask]
        data_dict["train"][NTP_ITEM_LABEL] = item_labels[valid_mask]
        data_dict["train"][NTP_TEXT_LABEL] = text_labels[valid_mask]
        data_dict["train"][NTP_GRAPH_LABEL] = graph_labels[valid_mask]

        logging.info(
            "DNTP labels: item=%s, text=%s, graph=%s (text_thr=%.4f, graph_thr=%.4f)",
            tuple(data_dict["train"][NTP_ITEM_LABEL].shape),
            tuple(data_dict["train"][NTP_TEXT_LABEL].shape),
            tuple(data_dict["train"][NTP_GRAPH_LABEL].shape),
            self.threshold_value["text"],
            self.threshold_value["graph"],
        )

        for split in ["valid", "test"]:
            split_df = inter_df[split]
            split_df[ITEM_SEQ_ID] = split_df[ITEM_SEQ_ID].apply(
                lambda x: x[-MAX_ITEM_SEQ_LEN:]
            )
            split_df[ITEM_SEQ_LEN] = split_df[ITEM_SEQ_ID].apply(len)
            item_seq = [
                torch.from_numpy(np.array(x)).long() for x in split_df[ITEM_SEQ_ID].values
            ]
            left_padded = [
                pad(seq, (MAX_ITEM_SEQ_LEN - len(seq), 0), value=0) for seq in item_seq
            ]
            data_dict[split][ITEM_SEQ_ID] = torch.stack(left_padded)
            data_dict[split][ITEM_SEQ_LEN] = torch.from_numpy(
                split_df[ITEM_SEQ_LEN].values
            ).long()
            data_dict[split][ITEM_ID] = torch.from_numpy(split_df[ITEM_ID].values).long()

        self.data_dict = data_dict

        logging.info(f"size of train: {len(self.data_dict['train'][ITEM_SEQ_ID])}")
        logging.info(f"size of valid: {len(self.data_dict['valid'][ITEM_ID])}")
        logging.info(f"size of test: {len(self.data_dict['test'][ITEM_ID])}")
        logging.info("Finish reading data.")
