# -*- coding: UTF-8 -*-
"""
SwingReader
===========

Loads sequence recommendation data (train/valid/test) and pre-generated Swing item-item
neighbors from parquet. Outputs tensors ready for PyTorch
datasets plus a dict `swing_topk_neighbours[item] -> List[(nbr, score)]`.

Main steps
----------
1. Read item & interaction CSVs, map original IDs to internal IDs, build feature dicts.
2. Pad / truncate user histories to MAX_ITEM_SEQ_LEN (left padding).
3. Load Swing neighbors from a pre-generated parquet graph.
4. Provide quick sanity evaluation of Swing neighbors (hit rate over last K item2item).
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import ast
from torch.nn.functional import pad

from utils import utils
from utils.constants import *
from helpers.BaseReader import BaseReader


class SwingReader(BaseReader):
    """
    Data reader + Swing preprocessor.

    Args (from CLI):
        path:            root folder of processed CSVs.
        dataset:         dataset name (subfolder under `path`).
        sep:             CSV separator.
        swing_topk:      keep top-k neighbors per item.
        swing_parquet_path:
                         optional path to pre-generated Swing parquet.

    Attributes set:
        data_dict:  dict split->tensor fields (ITEM_ID, ITEM_SEQ_ID, ITEM_SEQ_LEN)
        item_id2feat/item_feat_num: item feature maps and vocab sizes
        swing_topk_neighbours: dict[int] -> List[(nbr_id, weight)]
        n_users, n_items: cardinalities incl. PAD=0
    """

    @staticmethod
    def parse_data_args(parser):
        parser.add_argument(
            "--path",
            type=str,
            default="processed_llo_graph",
            help="Input data dir."
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="Software",
            help="Choose a dataset."
        )
        parser.add_argument(
            "--sep",
            type=str,
            default=",",
            help="sep of csv file."
        )
        parser.add_argument(
            "--swing_topk",
            type=int,
            default=100,
            help="Max neighbors per item."
        )
        parser.add_argument(
            "--swing_parquet_path",
            type=str,
            default="",
            help="Optional path to pre-generated swing parquet with columns: item_id, neighbors, weights. "
                 "If empty, defaults to {path}/{dataset}/graph.",
        )
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset

        # swing parquet parameters
        self.swing_topk = args.swing_topk
        self.swing_parquet_path = args.swing_parquet_path

        self.swing_topk_neighbours = None
        self._read_data()

    def test_swing(self, swing, data_dict, trigger_num=5, swing_topk_test=20):
        """
        Quick-and-dirty hit-rate check: does any of the last `trigger_num` items'
        neighbors contain the true target?

        Parameters
        ----------
        swing : dict[int] -> List[(nbr, score)]
        data_dict : split tensors
        trigger_num : int
        swing_topk_test : int
        """
        test_seqs = data_dict[ITEM_SEQ_ID][:, -trigger_num:]
        targets   = data_dict[ITEM_ID]
        score = 0.0
        not_in = 0

        for idx in range(test_seqs.size(0)):
            seq    = test_seqs[idx]
            target = int(targets[idx])
            hit = False
            for item in seq.tolist():
                if item not in swing:
                    not_in += 1
                    continue
                neighbors = [nbr for nbr, _ in swing[item][:swing_topk_test]]
                if target in neighbors:
                    score += 1
                    hit = True
                    break
        
        logging.info(f"For swing_topk_test={swing_topk_test}\n Score: {score / test_seqs.size(0)}")
        logging.info(f"No Swing neighbors count: {not_in}")

    @staticmethod
    def _parse_list_like(value):
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
            if "," in text:
                return [v.strip() for v in text.split(",") if v.strip()]
            return [v for v in text.split() if v]
        return list(value)

    def _load_swing_from_parquet(self, orig_item_id2item_id):
        parquet_path = self.swing_parquet_path
        if not parquet_path:
            parquet_path = os.path.join(self.prefix, self.dataset, "graph")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Swing parquet file not found: {parquet_path}")
        logging.info(f'Loading pre-generated swing neighbors from "{parquet_path}"')
        
        swing_df = pd.read_parquet(parquet_path)
        required_cols = {"item_id", "neighbors", "weights"}
        missing = required_cols - set(swing_df.columns)
        if missing:
            raise ValueError(f"Swing parquet must contain columns {sorted(required_cols)}. Missing: {sorted(missing)}")

        swing_neighbors = {}
        dropped_items = 0
        dropped_neighbors = 0
        
        for row in swing_df[["item_id", "neighbors", "weights"]].itertuples(index=False, name=None):
            orig_item_id, raw_neighbors, raw_weights = row
            item_id = orig_item_id2item_id.get(orig_item_id)
            if item_id is None:
                dropped_items += 1
                continue

            neighbors = self._parse_list_like(raw_neighbors)
            weights = self._parse_list_like(raw_weights)
            if len(neighbors) != len(weights):
                n = min(len(neighbors), len(weights))
                neighbors = neighbors[:n]
                weights = weights[:n]

            mapped = []
            for nbr_orig, weight in zip(neighbors, weights):
                nbr_id = orig_item_id2item_id.get(nbr_orig)
                if nbr_id is None:
                    dropped_neighbors += 1
                    continue
                try:
                    mapped.append((int(nbr_id), float(weight)))
                except (TypeError, ValueError):
                    continue

            if not mapped:
                continue
            mapped.sort(key=lambda x: x[1], reverse=True)
            swing_neighbors[int(item_id)] = mapped[: self.swing_topk]

        logging.info(
            "Loaded parquet swing neighbors for %d items. Dropped unknown item rows=%d, unknown neighbors=%d",
            len(swing_neighbors), dropped_items, dropped_neighbors,
        )
        return swing_neighbors

    def _read_data(self):        
        if self.dataset in [
            "CDs_and_Vinyl", 
            "Software",
            "Video_Games", 
            "Beauty_and_Personal_Care", 
            "Office_Products",
            "Arts_Crafts_and_Sewing",
            "Cell_Phones_and_Accessories",
            "Office_Products",
            "Toys_and_Games",
        ]:
            ORIG_ITEM_ID = "parent_asin"
            ITEM_DATA_COLUMN = [
                "parent_asin", "item_id", "text_emb",
            ]
            ITEM_FEAT_COLUMN = []

        self.item_feat_column = ITEM_FEAT_COLUMN

        logging.info(f'Reading data from "{self.prefix}", dataset = "{self.dataset}"')

        # -------- items --------
        item_df = pd.read_csv(
            os.path.join(self.prefix, self.dataset, f"{self.dataset}.item.csv"),
            sep=self.sep,
            usecols=ITEM_DATA_COLUMN,
        ).reset_index(drop=True)

        self.orig_item_id2item_id = item_df.set_index(ORIG_ITEM_ID)[ITEM_ID].to_dict()

        item_df.drop(columns=[ORIG_ITEM_ID], inplace=True)
        self.item_id2text_emb = item_df.set_index(ITEM_ID)["text_emb"].to_dict()
        item_df.drop(columns=["text_emb"], inplace=True)

        for feat in ITEM_FEAT_COLUMN:
            if feat.endswith("seq_id"):
                item_df[feat] = item_df[feat].apply(eval)

        self.item_id2feat = item_df.set_index(ITEM_ID).to_dict(orient="index")
        self.item_feat_num = {
            feat: len(set(item_df[feat].dropna().explode())) for feat in ITEM_FEAT_COLUMN
        }
        logging.info(f"item_feat_num: {self.item_feat_num}")

        # -------- interactions --------
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
            split_df[ITEM_ID] = split_df[ORIG_ITEM_ID].map(self.orig_item_id2item_id)
            split_df[ITEM_SEQ_ID] = split_df[ITEM_SEQ].apply(
                lambda x: [self.orig_item_id2item_id[iid] for iid in x]
            )
            user_set.update(split_df[USER_ID].tolist())
            n_entry += len(split_df)
            inter_df[key] = split_df

        logging.info("Counting dataset statistics...")
        
        self.n_users = len(user_set) + 1
        self.n_items = item_df[ITEM_ID].max() + 1
        del user_set

        logging.info(f'"# user": {self.n_users-1}, "# item": {self.n_items-1}, "# entry": {n_entry}')

        # -------- tensors --------
        data_dict = {key: dict() for key in ["train", "valid", "test"]}
        for split in ["train", "valid", "test"]:
            split_df = inter_df[split]
            split_df[ITEM_SEQ] = split_df[ITEM_SEQ].apply(lambda x: x[-MAX_ITEM_SEQ_LEN:])
            data_dict[split][ITEM_ID] = torch.from_numpy(split_df[ITEM_ID].values).long()
            item_seq = [torch.from_numpy(np.array(x)).long() for x in split_df[ITEM_SEQ_ID].values]
            left_padded = [pad(seq, (MAX_ITEM_SEQ_LEN - len(seq), 0), value=0) for seq in item_seq]
            data_dict[split][ITEM_SEQ_ID] = torch.stack(left_padded)
            data_dict[split][ITEM_SEQ_LEN] = torch.from_numpy(split_df[ITEM_SEQ_LEN].values).long()

        self.data_dict = data_dict
        del data_dict

        logging.info(f"size of train: {len(self.data_dict['train'][ITEM_ID])}")
        logging.info(f"size of valid: {len(self.data_dict['valid'][ITEM_ID])}")
        logging.info(f"size of test: {len(self.data_dict['test'][ITEM_ID])}")
        logging.info("Finish reading data.")

        # -------------------- Swing preprocessing --------------------
        self.swing_topk_neighbours = self._load_swing_from_parquet(self.orig_item_id2item_id)

        # quick tests
        for split in ["train", "valid", "test"]:
            logging.info(f"Test swing tables on {split} set. ")
            self.test_swing(self.swing_topk_neighbours, self.data_dict[split], trigger_num=1, swing_topk_test=20)
            self.test_swing(self.swing_topk_neighbours, self.data_dict[split], trigger_num=1, swing_topk_test=50)
            self.test_swing(self.swing_topk_neighbours, self.data_dict[split], trigger_num=1, swing_topk_test=100)      
        
        logging.info("--- Finish swing preprocessing ---")