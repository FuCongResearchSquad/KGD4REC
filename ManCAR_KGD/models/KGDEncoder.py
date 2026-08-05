# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn

from models.BaseModel import BaseModel
from utils.constants import *
import utils.layers_kgd as layers_kgd
import utils.wrappers_kgd as wrappers_kgd


class KGDEncoder(BaseModel):
    reader = "PretrainReader"
    runner = "BaseRunner"

    @staticmethod
    def parse_model_args(parser):
        parser = BaseModel.parse_model_args(parser)
        parser.add_argument(
            "--emb_size",
            type=int,
            default=256,
            help="Size of embeddings",
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=2,
            help="Number of transformer layers",
        )
        parser.add_argument(
            "--num_heads",
            type=int,
            default=2,
            help="Number of attention heads",
        )
        parser.add_argument(
            "--inner_size",
            type=int,
            default=300,
            help="FFN inner size",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.3,
            help="Dropout probability",
        )
        parser.add_argument(
            "--hidden_act",
            type=str,
            default="gelu",
            help="Activation function",
        )
        parser.add_argument(
            "--rms_norm_eps",
            type=float,
            default=1e-12,
            help="RMS normalization epsilon",
        )
        parser.add_argument(
            "--ntp_item_weight",
            type=float,
            default=1.0,
            help="Weight for the NTP item (next-item) loss term.",
        )
        parser.add_argument(
            "--ntp_text_weight",
            type=float,
            default=1.0,
            help="Weight for the NTP text-similarity loss term.",
        )
        parser.add_argument(
            "--ntp_graph_weight",
            type=float,
            default=1.0,
            help="Weight for the NTP graph-similarity loss term.",
        )
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.item_num = corpus.n_items
        self.emb_size = args.emb_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.inner_size = args.inner_size
        self.dropout = args.dropout
        self.hidden_act = args.hidden_act
        self.rms_norm_eps = args.rms_norm_eps
        self.ntp_item_weight = args.ntp_item_weight
        self.ntp_text_weight = args.ntp_text_weight
        self.ntp_graph_weight = args.ntp_graph_weight

        self._define_params()
        self.apply(self.init_weights)
        self.all_item_embs = None

    def _define_params(self):
        self.item_id_emb = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_ITEM_SEQ_LEN + 1, self.emb_size, padding_idx=0)

        trm_encoder = layers_kgd.TransformerEncoderv2(
            n_layers=self.num_layers,
            n_heads=self.num_heads,
            hidden_size=self.emb_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.dropout,
            attn_dropout_prob=self.dropout,
            hidden_act=self.hidden_act,
            rms_norm_eps=self.rms_norm_eps,
        )

        self.encoder = wrappers_kgd.CausalEncoderWrapper(
            transformer=trm_encoder,
            hidden_size=self.emb_size,
            dropout=self.dropout,
        )

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, feed_dict, epoch=0, stage="train"):
        item_seq_ids = feed_dict[ITEM_SEQ_ID]
        item_seq_len = feed_dict[ITEM_SEQ_LEN]

        padding_mask = item_seq_ids != 0
        valid_pos_ids = torch.cumsum(padding_mask.long(), dim=1)
        pos_ids = torch.where(padding_mask, valid_pos_ids, 0)

        item_embs = self.item_id_emb(item_seq_ids) + self.pos_emb(pos_ids)
        hidden = self.encoder(item_embs, item_seq_len)

        feed_dict["hidden"] = hidden
        return feed_dict

    def _safe_ce(self, logits, labels):
        """CrossEntropyLoss that returns 0 when all labels are ignored."""
        if (labels != -100).any():
            return self.loss_fct(logits, labels)
        return logits.new_tensor(0.0)

    def loss(self, out_dict):
        hidden = out_dict["hidden"]
        logits = torch.matmul(hidden, self.item_id_emb.weight.T)
        logits_flat = logits.view(-1, self.item_num)

        item_loss = self._safe_ce(logits_flat, out_dict[NTP_ITEM_LABEL].view(-1))
        text_loss = self._safe_ce(logits_flat, out_dict[NTP_TEXT_LABEL].view(-1))
        graph_loss = self._safe_ce(logits_flat, out_dict[NTP_GRAPH_LABEL].view(-1))

        return (
            self.ntp_item_weight * item_loss
            + self.ntp_text_weight * text_loss
            + self.ntp_graph_weight * graph_loss
        )

    @torch.no_grad()
    def encode_all_items(self, batch_size=None):
        self.all_item_embs = self.item_id_emb.weight.data

    @torch.no_grad()
    def inference(self, feed_dict):
        out = self.forward(feed_dict, stage="infer")
        hidden = out["hidden"]
        seq_embs = hidden[:, -1, :]
        logits = torch.matmul(seq_embs, self.all_item_embs.T)
        return {"prediction": logits}

    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = {
                ITEM_SEQ_ID: self.data[ITEM_SEQ_ID][index],
                ITEM_SEQ_LEN: self.data[ITEM_SEQ_LEN][index],
            }
            for key in (NTP_ITEM_LABEL, NTP_TEXT_LABEL, NTP_GRAPH_LABEL):
                if key in self.data:
                    feed_dict[key] = self.data[key][index]
            if ITEM_ID in self.data:
                feed_dict[ITEM_ID] = self.data[ITEM_ID][index]
            return feed_dict
