# -*- coding: UTF-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ManCAR import ManCAR
from utils import layers_kgd
from utils import wrappers_kgd
from utils.constants import *


class ManCAR_KGD(ManCAR):
    reader = "SwingReader"
    runner = "BaseRunner"

    @staticmethod
    def parse_model_args(parser):
        parser = ManCAR.parse_model_args(parser)
        parser.add_argument(
            "--kgd_learner_num_layers",
            type=int,
            default=2,
            help="Number of Transformer decoder layers for reasoning.",
        )
        parser.add_argument(
            "--acr_rank",
            type=int,
            default=64,
            help="Rank of the ACR item-side low-rank factor.",
        )
        parser.add_argument(
            "--acr_weight_decay",
            type=float,
            default=0.5,
            help="Weight decay applied exclusively to acr_item_factor.",
        )
        parser.add_argument(
            "--acr_use_scale",
            type=int,
            default=0,
            help="ACR task scale.",
        )
        parser.add_argument(
            "--pretrained_encoder_path",
            type=str,
            default="",
            help="KGD encoder checkpoint path for warm-start initialization.",
        )
        parser.add_argument(
            "--pretrain_init",
            type=int,
            default=1,
            help="Whether to load pretrained_encoder_path. 1=load, 0=keep initialization.",
        )
        parser.add_argument(
            "--freeze_pretrained_encoder",
            type=int,
            default=1,
            help="Freeze pretrained encoder weights.",
        )
        parser.add_argument(
            "--orth_weight",
            type=float,
            default=0.2,
            help="Global weight of the per-position orth regularizer.",
        )
        parser.add_argument(
            "--orth_history_weight",
            type=float,
            default=1.0,
            help="Relative weight of the history merge-point orth term.",
        )
        parser.add_argument(
            "--orth_context_weight",
            type=float,
            default=1.0,
            help="Relative weight of the context merge-point orth term.",
        )
        parser.add_argument(
            "--orth_detach_main",
            type=int,
            default=1,
            help="Detach main item embedding branch when computing orth loss.",
        )
        parser.add_argument(
            "--orth_eps",
            type=float,
            default=1e-8,
            help="Numerical epsilon for L2 normalization in orth loss.",
        )
        return parser

    def __init__(self, args, corpus):
        self.kgd_learner_num_layers = args.kgd_learner_num_layers
        self.acr_rank = args.acr_rank
        self.acr_weight_decay = float(getattr(args, "acr_weight_decay", 0.5))
        self.acr_use_scale = int(getattr(args, "acr_use_scale", 0))
        self.pretrain_init = getattr(args, "pretrain_init", 1)
        self.pretrain_init_modules = ("item_id_emb,pos_emb,model.encoder,model.encoder_norm")
        self.freeze_pretrained_encoder = getattr(args, "freeze_pretrained_encoder", 1)
        self.pretrained_encoder_path = getattr(args, "pretrained_encoder_path", "")
        self.orth_weight = float(getattr(args, "orth_weight", 0.5))
        self.orth_history_weight = float(getattr(args, "orth_history_weight", 1.0))
        self.orth_context_weight = float(getattr(args, "orth_context_weight", 1.0))
        self.orth_detach_main = int(getattr(args, "orth_detach_main", 1))
        self.orth_eps = float(getattr(args, "orth_eps", 1e-8))

        super().__init__(args, corpus)
        
        self.item_id2feat = corpus.item_id2feat
        self.all_item_embs = None

        if self.pretrain_init == 1:
            if not self.pretrained_encoder_path:
                raise ValueError("pretrained_encoder_path must be provided when pretrain_init=1",)
            self._load_pretrain_init(self.pretrained_encoder_path)
        self._apply_freeze_strategy()

    def _define_params(self, corpus):
        self.item_id_emb = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_ITEM_SEQ_LEN + 1, self.emb_size, padding_idx=0)

        self.feat_emb = nn.ModuleDict()
        for feat in corpus.item_feat_column:
            self.feat_emb[feat] = nn.Embedding(
                corpus.item_feat_num[feat] + 1, self.emb_size, padding_idx=0,
            )

        encoder = layers_kgd.TransformerEncoderv2(
            n_layers=self.num_layers,
            n_heads=self.num_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.dropout,
            attn_dropout_prob=self.dropout,
            hidden_act=self.hidden_act,
            rms_norm_eps=self.rms_norm_eps,
        )
        decoder = layers_kgd.KGDReadOnlyLearner(
            n_layers=self.kgd_learner_num_layers,
            n_heads=self.num_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.dropout,
            attn_dropout_prob=self.dropout,
            hidden_act=self.hidden_act,
            rms_norm_eps=self.rms_norm_eps,
        )
        self.model = wrappers_kgd.ManCARKGDWrapper(
            encoder=encoder,
            decoder=decoder,
            hidden_size=self.hidden_size,
            reason_step=self.reason_step,
            noise_factor=self.noise,
            dropout=self.dropout,
            layer_norm_eps=self.rms_norm_eps,
        )

        self.acr_item_factor = nn.Embedding(self.item_num, self.acr_rank, padding_idx=0,)
        self.acr_residual_proj = nn.Linear(self.acr_rank, self.hidden_size,)
        self.acr_raw_scale = nn.Parameter(torch.tensor(1e-3))

        self.loss_fct = nn.CrossEntropyLoss()

    class Dataset(ManCAR.Dataset):
        """Use ManCAR supervision and add mixed-context neighbour ids."""

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)

            item_seq = feed_dict[ITEM_SEQ_ID]
            non_pad = [iid for iid in item_seq.tolist() if iid != 0]
            trigger_items = non_pad[-self.trigger_num:]

            context_ids = []
            for item_id in trigger_items:
                neighbours = [nid for nid, _ in self.swing_neighbors.get(item_id, [])]
                neighbours = neighbours[: self.item_per_trigger]
                if len(neighbours) < self.item_per_trigger:
                    neighbours += [0] * (self.item_per_trigger - len(neighbours))
                context_ids.extend(neighbours)

            total_len = self.trigger_num * self.item_per_trigger
            if len(context_ids) < total_len:
                context_ids += [0] * (total_len - len(context_ids))

            feed_dict["context_prompt_ids"] = torch.tensor(context_ids, dtype=torch.long,)
            
            feed_dict.pop("prompt_ids", None)
            feed_dict.pop("trigger_ids", None)
            for key in list(feed_dict.keys()):
                if key.startswith("prompt_"):
                    del feed_dict[key]
            
            return feed_dict

    def _get_history_item_embs(self, feed_dict):
        item_seq_ids = feed_dict[ITEM_SEQ_ID]
        item_feat_embs = [self.item_id_emb(item_seq_ids)]
        for feat in self.feat_emb:
            feat_ids = feed_dict[f"seq_{feat}"]
            feat_emb = self.feat_emb[feat](feat_ids)
            feat_emb = self.avg_feat_emb(feat_emb, feat_ids)
            item_feat_embs.append(feat_emb)
        return torch.sum(torch.stack(item_feat_embs, dim=2), dim=2)

    def _get_acr_residual(self, item_ids):
        return self.acr_residual_proj(self.acr_item_factor(item_ids))

    def _get_acr_scale(self):
        if not self.acr_use_scale:
            return self.acr_raw_scale.new_tensor(1.0)
        return 1.0 + F.relu(self.acr_raw_scale)

    def _orth_enabled(self) -> bool:
        if self.orth_weight == 0.0:
            return False
        return (self.orth_history_weight != 0.0) or (self.orth_context_weight != 0.0)

    def _masked_sq_cos_sim(
        self,
        main_embs: torch.Tensor,
        aux_embs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        main_branch = F.normalize(main_embs, dim=-1, eps=self.orth_eps)
        aux_branch = F.normalize(aux_embs, dim=-1, eps=self.orth_eps)

        if self.orth_detach_main:
            main_branch = main_branch.detach()

        cos_sim = (main_branch * aux_branch).sum(dim=-1)
        valid_mask = mask.float()
        denom = valid_mask.sum().clamp_min(1.0)
        return (cos_sim.pow(2) * valid_mask).sum() / denom

    def _ensure_all_item_embs(self):
        if self.all_item_embs is not None:
            return
        was_training = self.training
        self.encode_all_items()
        if was_training:
            self.train()

    def forward(self, feed_dict: dict, epoch=0, stage="train") -> dict:
        item_seq_ids, item_seq_len = feed_dict[ITEM_SEQ_ID], feed_dict[ITEM_SEQ_LEN]
        batch_size = item_seq_ids.size(0)
        device = item_seq_ids.device
        self._ensure_all_item_embs()

        hist_padding_mask = item_seq_ids != 0
        valid_pos_ids = torch.cumsum(hist_padding_mask.long(), dim=1)
        pos_ids = torch.where(hist_padding_mask, valid_pos_ids, 0)
        pos_embs = self.pos_emb(pos_ids)

        hist_main_embs = self._get_history_item_embs(feed_dict)
        hist_acr_residual = self._get_acr_residual(item_seq_ids)
        acr_scale = self._get_acr_scale()
        input_embs = acr_scale * hist_main_embs + hist_acr_residual + pos_embs

        context_prompt_ids = feed_dict["context_prompt_ids"]
        context_memory = (
            acr_scale * self.item_id_emb(context_prompt_ids)
            + self._get_acr_residual(context_prompt_ids)
        )

        output = self.model(
            input_embs,
            item_seq_len,
            self.all_item_embs,
            noise_factor=self.noise,
            context_memory=context_memory,
            context_prompt_ids=context_prompt_ids,
        )

        seq_embs = output[:batch_size, -1, :]
        if stage == "all_steps":
            seq_embs = output

        feed_dict["seq_embs"] = seq_embs
        feed_dict["seq_output"] = output
        
        feed_dict["prompt_latent"] = torch.empty(
            batch_size, 0, self.hidden_size, device=device,
        )
        feed_dict["trigger_ids"] = torch.zeros(
            batch_size, 0, dtype=torch.long, device=device,
        )
        
        if stage == "train" and self._orth_enabled():
            if self.orth_history_weight != 0.0:
                feed_dict["orth_hist_main"] = hist_main_embs
                feed_dict["orth_hist_aux"] = hist_acr_residual
                feed_dict["orth_hist_mask"] = hist_padding_mask
            if self.orth_context_weight != 0.0:
                feed_dict["orth_ctx_main"] = self.item_id_emb(context_prompt_ids)
                feed_dict["orth_ctx_aux"] = self._get_acr_residual(context_prompt_ids)
                feed_dict["orth_ctx_mask"] = context_prompt_ids != 0

        feed_dict["epoch"] = epoch
        
        return feed_dict

    def _add_orth_loss(self, base_loss: torch.Tensor, out_dict: dict) -> torch.Tensor:
        if not self._orth_enabled():
            return base_loss

        orth_terms = []
        total_orth = base_loss.new_zeros(())

        if "orth_hist_main" in out_dict and self.orth_history_weight != 0.0:
            hist_loss = self._masked_sq_cos_sim(
                out_dict["orth_hist_main"],
                out_dict["orth_hist_aux"],
                out_dict["orth_hist_mask"],
            )
            total_orth = total_orth + self.orth_history_weight * hist_loss
            out_dict["orth_history_loss"] = hist_loss.detach()
            orth_terms.append(hist_loss)
        if "orth_ctx_main" in out_dict and self.orth_context_weight != 0.0:
            ctx_loss = self._masked_sq_cos_sim(
                out_dict["orth_ctx_main"],
                out_dict["orth_ctx_aux"],
                out_dict["orth_ctx_mask"],
            )
            total_orth = total_orth + self.orth_context_weight * ctx_loss
            out_dict["orth_context_loss"] = ctx_loss.detach()
            orth_terms.append(ctx_loss)

        if not orth_terms:
            return base_loss
        weighted_orth = self.orth_weight * total_orth
        out_dict["orth_loss"] = total_orth.detach()
        out_dict["orth_weighted_loss"] = weighted_orth.detach()
        return base_loss + weighted_orth

    def loss(self, out_dict: dict) -> torch.Tensor:
        base_loss = super().loss(out_dict)
        return self._add_orth_loss(base_loss, out_dict)

    def _load_pretrain_init(self, model_path):
        ckpt = torch.load(model_path, map_location=self.device)
        ckpt_state = (
            ckpt["state_dict"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt
            else ckpt
        )
        if not isinstance(ckpt_state, dict):
            raise ValueError(f"Invalid checkpoint format at {model_path}")

        selected_modules = [
            m.strip() for m in self.pretrain_init_modules.split(",") if m.strip()
        ]
        prefix_map = {
            "encoder.transformer.": "model.encoder.",
            "encoder.LayerNorm.": "model.encoder_norm.",
        }

        target_state = self.state_dict()
        remapped = {}
        skipped_shape, skipped_missing = [], []

        for src_key, src_weight in ckpt_state.items():
            candidate_keys = [src_key]
            for src_prefix, dst_prefix in prefix_map.items():
                if src_key.startswith(src_prefix):
                    candidate_keys.append(dst_prefix + src_key[len(src_prefix):])
            for dst_key in candidate_keys:
                if not any(dst_key.startswith(mod) for mod in selected_modules):
                    continue
                if dst_key not in target_state:
                    skipped_missing.append((src_key, dst_key))
                    continue
                if target_state[dst_key].shape != src_weight.shape:
                    skipped_shape.append((src_key, dst_key))
                    continue
                remapped[dst_key] = src_weight
                break

        load_result = self.load_state_dict(remapped, strict=False)
        logging.info(
            "Pretrain init from %s: \n"
            "-- loaded=%d,\n"
            "-- skipped_shape=%d,\n"
            "-- skipped_missing=%d,\n"
            "-- missing_after_load=%d,\n"
            "-- unexpected_after_load=%d",
            model_path,
            len(remapped),
            len(skipped_shape),
            len(skipped_missing),
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )

    @staticmethod
    def _freeze_module_params(module, name):
        count = 0
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False
                count += 1
        logging.info("Freeze [%s]: %d parameter tensors frozen", name, count)

    @staticmethod
    def _unfreeze_module_params(module, name):
        count = 0
        for param in module.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                count += 1
        if count:
            logging.info("Unfreeze [%s]: %d parameter tensors unfrozen", name, count)

    def _apply_freeze_strategy(self):
        if self.freeze_pretrained_encoder:
            self._freeze_module_params(self.item_id_emb, "item_id_emb")
            self._freeze_module_params(self.pos_emb, "pos_emb")
            self._freeze_module_params(self.model.encoder, "model.encoder")
            self._freeze_module_params(self.model.encoder_norm, "model.encoder_norm")
            self._freeze_module_params(self.model.encoder_dropout, "model.encoder_dropout")

        self._unfreeze_module_params(self.acr_item_factor, "acr_item_factor")
        self._unfreeze_module_params(self.acr_residual_proj, "acr_residual_proj")

        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = total - frozen
        logging.info(
            "GLDP freeze summary: total_params=%d, frozen=%d (%.1f%%), "
            "trainable=%d (ACR branch always trainable)",
            total, frozen, 100.0 * frozen / max(total, 1), trainable,
        )

    def customize_parameters(self):
        acr_factor_param_ids = {id(p) for p in self.acr_item_factor.parameters()}

        acr_factor_params = []
        other_weight_params = []
        bias_params = []

        for name, param in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if id(param) in acr_factor_param_ids:
                acr_factor_params.append(param)
            elif "bias" in name:
                bias_params.append(param)
            else:
                other_weight_params.append(param)

        def _numel(params):
            return sum(int(p.numel()) for p in params)

        logging.info(
            "GLDP param groups: acr_item_factor=%d tensors/%d params "
            "(weight_decay=%.3e), other_weight=%d tensors/%d params, "
            "bias=%d tensors/%d params (weight_decay=0)",
            len(acr_factor_params), _numel(acr_factor_params),
            self.acr_weight_decay,
            len(other_weight_params), _numel(other_weight_params),
            len(bias_params), _numel(bias_params),
        )

        return [
            {"params": acr_factor_params, "weight_decay": self.acr_weight_decay},
            {"params": other_weight_params},
            {"params": bias_params, "weight_decay": 0},
        ]

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path

        ckpt_state = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt_state, dict) and "state_dict" in ckpt_state:
            ckpt_state = ckpt_state["state_dict"]

        load_result = self.load_state_dict(ckpt_state, strict=False)
        logging.info(
            "load_model: missing=%d, unexpected=%d",
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )
