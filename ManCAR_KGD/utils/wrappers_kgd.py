import torch
import torch.nn as nn

from utils.constants import *
from utils.layers_kgd import ThoughtAdapter


class CausalEncoderWrapper(nn.Module):
    """
    CausalEncoderWrapper is a wrapper for causal transformer model to encode sequence.

    Args
        transformer (nn.Module): the transformer
        hidden_size (int): the hidden size of the transformer model

    Returns:
        all_outputs (torch.Tensor): the output of the sequence
    """

    def __init__(self, transformer, hidden_size, dropout, layer_norm_eps=1e-12):
        super(CausalEncoderWrapper, self).__init__()
        self.transformer = transformer
        self.hidden_size = hidden_size
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)

    def _prepare_attention_mask(self, batch_size, seq_len, device, padding_mask):
        mask = torch.ones((batch_size, 1, seq_len, seq_len), device=device)
        mask = torch.tril(mask)
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def _prepare_padding_mask(self, input_lens, device):
        input_lens = input_lens
        batch_size = len(input_lens)
        max_item_seq_len = MAX_ITEM_SEQ_LEN
        padding_mask = (
            torch.arange(max_item_seq_len)
            .unsqueeze(0)
            .expand(batch_size, max_item_seq_len)
            .to(device)
        )
        padding_mask = padding_mask < (max_item_seq_len - input_lens.unsqueeze(1))
        return padding_mask.unsqueeze(1).unsqueeze(2)

    def forward(self, input_embs, input_lens):
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device

        padding_mask = self._prepare_padding_mask(input_lens, device)
        attention_mask = self._prepare_attention_mask(
            batch_size, seq_len, device, padding_mask
        )

        input_embs = self.LayerNorm(input_embs)
        input_embs = self.dropout(input_embs)

        outputs = self.transformer(
            input_embs, attention_mask, output_all_encoded_layers=True
        )
        last_later_outputs = outputs[0][-1]

        return last_later_outputs  # [batch_size, seq_len, hidden_size]


class ManCARKGDWrapper(nn.Module):
    """GLDP encoder-decoder wrapper with mixed history/context attention.

    This class intentionally does not inherit from ``ManCARv2NextWrapper`` so
    the final GLDP path is self-contained in this module.
    """

    def __init__(self, encoder, decoder, hidden_size, reason_step,
                 noise_factor=0.0, dropout=0.5, layer_norm_eps=1e-12):
        super(ManCARKGDWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_encoder_layers = len(encoder.layers)
        self.n_decoder_layers = len(decoder.layers)
        self.hidden_size = hidden_size
        self.reason_step = reason_step
        self.noise_factor = noise_factor

        self.encoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.encoder_dropout = nn.Dropout(dropout)
        self.decoder_dropout = nn.Dropout(dropout)

        if reason_step > 0:
            self.reason_pos_emb = nn.Embedding(reason_step, hidden_size)
        self.adapter_norm = ThoughtAdapter(hidden_size)

    def _prepare_encoder_padding_mask(self, input_lens, device):
        batch_size = len(input_lens)
        seq_len = MAX_ITEM_SEQ_LEN
        idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pad = idx < (seq_len - input_lens.unsqueeze(1))
        return pad.unsqueeze(1).unsqueeze(2)

    def _prepare_encoder_mask(self, batch_size, seq_len, device, padding_mask):
        mask = torch.ones((batch_size, 1, seq_len, seq_len), device=device)
        mask = torch.tril(mask)
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def _prepare_decoder_self_mask(self, batch_size, step, device):
        seq_len = step + 1
        mask = torch.ones((seq_len, seq_len), device=device)
        mask = torch.tril(mask)
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        return mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len)

    def _prepare_mixed_cross_attn_mask(
        self,
        batch_size,
        encoder_seq_len,
        input_lens,
        context_prompt_ids,
        query_len,
        device,
    ):
        enc_idx = torch.arange(encoder_seq_len, device=device).unsqueeze(0)
        enc_idx = enc_idx.expand(batch_size, -1)
        enc_pad = enc_idx < (encoder_seq_len - input_lens.unsqueeze(1))

        ctx_pad = context_prompt_ids.eq(0)
        combined_pad = torch.cat([enc_pad, ctx_pad], dim=1)
        combined_pad = combined_pad.unsqueeze(1).unsqueeze(2)

        kv_len = combined_pad.size(-1)
        mask = torch.zeros((batch_size, 1, query_len, kv_len), device=device)
        return mask.masked_fill(combined_pad.expand(-1, -1, query_len, -1), -1e10)

    def forward(
        self,
        input_embs,
        input_lens,
        item_embs,
        noise_factor=None,
        context_memory=None,
        context_prompt_ids=None,
    ):
        if context_memory is None or context_prompt_ids is None:
            raise ValueError("ManCARKGDWrapper requires context memory.")

        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device
        if noise_factor is None:
            noise_factor = self.noise_factor

        enc_padding = self._prepare_encoder_padding_mask(input_lens, device)
        enc_mask = self._prepare_encoder_mask(batch_size, seq_len, device, enc_padding)

        enc_inp = self.encoder_dropout(self.encoder_norm(input_embs))
        encoder_output, _, _, _ = self.encoder(enc_inp, enc_mask)
        seed = encoder_output[:, -1:, :]

        repeat_batch = (noise_factor > 0.0) + 1
        enc_out = encoder_output.repeat(repeat_batch, 1, 1)
        lens_rep = input_lens.repeat(repeat_batch)
        seed = seed.repeat(repeat_batch, 1, 1)
        ctx_mem = context_memory.repeat(repeat_batch, 1, 1)
        ctx_ids = context_prompt_ids.repeat(repeat_batch, 1)

        mixed_mask = self._prepare_mixed_cross_attn_mask(
            batch_size * repeat_batch,
            seq_len,
            lens_rep,
            ctx_ids,
            1,
            device,
        )

        if self.reason_step == 0:
            self_mask = self._prepare_decoder_self_mask(
                batch_size * repeat_batch, 0, device
            )
            dec_inp = self.decoder_dropout(self.decoder_norm(seed))
            h, _ = self.decoder(dec_inp, self_mask, enc_out, ctx_mem, mixed_mask)
            return h

        decoder_kv = [None] * self.n_decoder_layers
        all_outputs, all_noise_outputs = [], []

        for step in range(self.reason_step + 1):
            self_mask = self._prepare_decoder_self_mask(
                batch_size * repeat_batch, step, device
            )

            dec_inp = self.decoder_dropout(self.decoder_norm(seed))
            h, decoder_kv = self.decoder(
                dec_inp, self_mask, enc_out, ctx_mem, mixed_mask, decoder_kv
            )

            step_h = h[:batch_size, -1:, :]
            all_outputs.append(step_h)
            adapted_h = self.adapter_norm(step_h, item_embs)

            if noise_factor > 0.0:
                all_noise_outputs.append(h[batch_size:, -1:, :])

            if step == self.reason_step:
                break

            new_pos = self.reason_pos_emb(
                torch.tensor([step], device=device)
            ).expand(batch_size, 1, -1)
            seed = adapted_h + new_pos

            if noise_factor > 0.0:
                noise = torch.randn_like(seed) * noise_factor
                seed = torch.cat([seed, seed + noise], dim=0)

        outputs = torch.cat(all_outputs, dim=1)
        if noise_factor > 0.0:
            outputs = torch.cat([outputs, torch.cat(all_noise_outputs, dim=1)], dim=0)

        return outputs
