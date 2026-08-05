# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import math
import copy
from torch.nn import functional as F

from utils.constants import *


class CrossMultiHeadAttention(nn.Module):
    """
    Cross Multi-head Attention layers, a attention score dropout layer is introduced.

    This layer performs cross-attention between two sequences, where queries come from
    one sequence and keys/values come from another sequence.

    Args:
        query_tensor (torch.Tensor): the query input tensor [batch_size, query_seq_len, hidden_size]
        key_value_tensor (torch.Tensor): the key-value input tensor [batch_size, kv_seq_len, hidden_size]
        attention_mask (torch.Tensor): the attention mask for cross attention [batch_size, 1, query_seq_len, kv_seq_len]

    Returns:
        hidden_states (torch.Tensor): the output of the cross multi-head attention layer
        kv_cache (dict): cached key-value pairs for inference optimization
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        rms_norm_eps,
    ):
        super(CrossMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        # Query projection from the first sequence
        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        # Key and Value projections from the second sequence
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.RMSNorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def _prepare_attention_mask(
        self, batch_size, query_seq_len, kv_seq_len, device, padding_mask
    ):
        mask = torch.zeros((batch_size, 1, query_seq_len, kv_seq_len), device=device)
        expanded_padding_mask = padding_mask.expand(-1, -1, query_seq_len, -1)
        mask = mask.masked_fill(expanded_padding_mask, -1e10)
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

    def forward(self, query_tensor, key_value_tensor, attention_mask):
        """
        Forward pass for cross multi-head attention.

        Args:
            query_tensor: [batch_size, query_seq_len, hidden_size]
            key_value_tensor: [batch_size, kv_seq_len, hidden_size]
            attention_mask: [batch_size, 1, query_seq_len, kv_seq_len] pre-computed mask
                            (0 = attend, -1e10 = block)
        """
        query = self.query_layer(query_tensor)

        # Generate key and value from key_value_tensor
        key = self.key_layer(key_value_tensor)
        value = self.value_layer(key_value_tensor)

        # Reshape for multi-head attention
        # query: [batch_size, num_heads, query_seq_len, head_size]
        query = self.transpose_for_scores(query).permute(0, 2, 1, 3)
        # key: [batch_size, num_heads, head_size, kv_seq_len]
        key = self.transpose_for_scores(key).permute(0, 2, 3, 1)
        # value: [batch_size, num_heads, kv_seq_len, head_size]
        value = self.transpose_for_scores(value).permute(0, 2, 1, 3)

        # Calculate attention scores
        # [batch_size, num_heads, query_seq_len, kv_seq_len]
        attention_scores = torch.matmul(query, key)
        attention_scores = attention_scores / self.sqrt_attention_head_size

        # Apply attention mask
        # attention_mask should be [batch_size, 1, query_seq_len, kv_seq_len]
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax to get attention probabilities
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # Apply attention to values
        # [batch_size, num_heads, query_seq_len, head_size]
        context_layer = torch.matmul(attention_probs, value)

        # Reshape back to original format
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Apply output projection and normalization
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        # Residual connection with query_tensor (not key_value_tensor)
        hidden_states = self.RMSNorm(hidden_states + query_tensor)

        return hidden_states


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        rms_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.RMSNorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, kv_cache=None):
        query = self.query_layer(input_tensor)
        key = self.key_layer(input_tensor)
        value = self.value_layer(input_tensor)

        if kv_cache is not None:
            key = torch.cat([kv_cache["key"], key], dim=1)
            value = torch.cat([kv_cache["value"], value], dim=1)
        else:
            kv_cache = {}

        kv_cache["key"] = key
        kv_cache["value"] = value

        query = self.transpose_for_scores(query).permute(0, 2, 1, 3)
        key = self.transpose_for_scores(key).permute(0, 2, 3, 1)
        value = self.transpose_for_scores(value).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
       
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)

        return hidden_states, kv_cache, attention_scores


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer
    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, rms_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.RMSNorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": F.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerEncoderLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask, kv_cache=None):
        attention_output, new_kv_cache, attention_scores = self.multi_head_attention(
            hidden_states, attention_mask, kv_cache
        )
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, new_kv_cache, attention_scores, attention_mask


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerEncoderLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        rms_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerEncoderLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            rms_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=False,
        kv_caches=None,
    ):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
            kv_caches (list): a list of key and value caches for each transformer layer
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        present_kv_caches = []
        attention_scores_list = []
        attention_mask_list = []

        for i, layer_module in enumerate(self.layer):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            if layer_kv_cache:
                layer_outputs, attention_scores, attention_mask = layer_module(
                    hidden_states[:, -1:, :], attention_mask, layer_kv_cache
                )
            else:
                layer_outputs, attention_scores, attention_mask = layer_module(
                    hidden_states, attention_mask, layer_kv_cache
                )
            hidden_states = layer_outputs[0]
            present_kv_caches.append(layer_outputs[1])
            attention_mask_list.append(attention_mask)
            attention_scores_list.append(attention_scores)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, present_kv_caches, attention_scores_list, attention_mask_list


class TransformerEncoderv2(nn.Module):
    r"""One TransformerEncoder consists of several TransformerEncoderLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        rms_norm_eps=1e-12,
    ):
        super(TransformerEncoderv2, self).__init__()
        layer = TransformerEncoderLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            rms_norm_eps,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attn_mask, *,
            kv_caches=None, output_all_encoded_layers=False, fp16_kv=False):

        use_half = torch.float16 if fp16_kv else hidden_states.dtype
        out_list = [] if output_all_encoded_layers else None
        new_kv   = []
        x = hidden_states                    # will hold “current sequence” slice
        
        attention_scores_list = []
        attention_mask_list = []
        
        for i, layer in enumerate(self.layers):
            kvc = None if kv_caches is None else kv_caches[i]

            # ---- 1. choose the slice we send to this layer ----
            if kvc is None:                  # full pass (step‑0)
                inp = x                      # [B, S, D]
            else:                            # incremental step
                inp = x[:, -1:, :]           # [B, 1, D]

            # ---- 2. run the layer (no checkpoint inside incremental) ----
            h, kv, attention_scores, attention_mask = layer(inp, attn_mask, kvc)

            # ---- 3. cast KV once, store back as dict ----
            if kv is not None:
                kv = {"key": kv["key"].to(use_half), "value": kv["value"].to(use_half)}
            new_kv.append(kv)

            # ---- 4. update running sequence only AFTER last layer ----
            if i == len(self.layers) - 1:
                x = torch.cat([x, h], dim=1) if kvc is not None else h
            else:
                x = x if kvc is not None else h   # keep same length inside block

            if output_all_encoded_layers:
                out_list.append(x)
            attention_mask_list.append(attention_mask)
            attention_scores_list.append(attention_scores)
        if output_all_encoded_layers:
            return out_list, new_kv
        return x, new_kv, attention_scores_list, attention_mask_list


class TransformerDecoderLayer(nn.Module):
    """Standard Transformer decoder layer: causal self-attention, cross-attention, FFN."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        rms_norm_eps,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, rms_norm_eps
        )
        self.cross_attention = CrossMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, rms_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            rms_norm_eps,
        )

    def forward(self, hidden_states, self_attn_mask, encoder_output, cross_attn_mask,
                self_kv_cache=None):
        attn_out, new_kv_cache, attn_scores = self.self_attention(
            hidden_states, self_attn_mask, self_kv_cache
        )
        cross_out = self.cross_attention(attn_out, encoder_output, cross_attn_mask)
        output = self.feed_forward(cross_out)
        return output, new_kv_cache


class TransformerDecoder(nn.Module):
    """Stack of TransformerDecoderLayers with per-layer self-attention KV cache."""

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        rms_norm_eps=1e-12,
    ):
        super(TransformerDecoder, self).__init__()
        layer = TransformerDecoderLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            rms_norm_eps,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, self_attn_mask, encoder_output, cross_attn_mask,
                kv_caches=None):
        new_kv = []
        x = hidden_states
        for i, layer in enumerate(self.layers):
            kvc = kv_caches[i] if kv_caches is not None else None
            inp = x[:, -1:, :] if kvc is not None else x
            h, kv = layer(inp, self_attn_mask, encoder_output, cross_attn_mask, kvc)
            new_kv.append(kv)
            x = h
        return x, new_kv


class HybridCrossAttention(nn.Module):
    """Cross-attention that jointly attends to encoder output and context memory."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        rms_norm_eps,
    ):
        super(HybridCrossAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)
        self.context_key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.context_value_layer = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.RMSNorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, query_tensor, encoder_kv, context_kv, attention_mask):
        query = self.query_layer(query_tensor)

        enc_key = self.key_layer(encoder_kv)
        enc_val = self.value_layer(encoder_kv)
        ctx_key = self.context_key_layer(context_kv)
        ctx_val = self.context_value_layer(context_kv)

        key = torch.cat([enc_key, ctx_key], dim=1)
        value = torch.cat([enc_val, ctx_val], dim=1)

        query = self.transpose_for_scores(query).permute(0, 2, 1, 3)
        key = self.transpose_for_scores(key).permute(0, 2, 3, 1)
        value = self.transpose_for_scores(value).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query, key)
        attention_scores = attention_scores / self.sqrt_attention_head_size

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + query_tensor)

        return hidden_states


class KGDReadOnlyLearnerLayer(nn.Module):
    """Decoder layer with mixed cross-attention over encoder and context memory."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        rms_norm_eps,
    ):
        super(KGDReadOnlyLearnerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, rms_norm_eps
        )
        self.cross_attention = HybridCrossAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, rms_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            rms_norm_eps,
        )

    def forward(
        self,
        hidden_states,
        self_attn_mask,
        encoder_output,
        context_memory,
        mixed_cross_mask,
        self_kv_cache=None,
    ):
        attn_out, new_kv_cache, attn_scores = self.self_attention(
            hidden_states, self_attn_mask, self_kv_cache
        )
        cross_out = self.cross_attention(
            attn_out, encoder_output, context_memory, mixed_cross_mask
        )
        output = self.feed_forward(cross_out)
        return output, new_kv_cache


class KGDReadOnlyLearner(nn.Module):
    """Stack of mixed decoder layers with per-layer self-attention KV cache."""

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        rms_norm_eps=1e-12,
    ):
        super(KGDReadOnlyLearner, self).__init__()
        layer = KGDReadOnlyLearnerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            rms_norm_eps,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        hidden_states,
        self_attn_mask,
        encoder_output,
        context_memory,
        mixed_cross_mask,
        kv_caches=None,
    ):
        new_kv = []
        x = hidden_states
        for i, layer in enumerate(self.layers):
            kvc = kv_caches[i] if kv_caches is not None else None
            inp = x[:, -1:, :] if kvc is not None else x
            h, kv = layer(
                inp,
                self_attn_mask,
                encoder_output,
                context_memory,
                mixed_cross_mask,
                self_kv_cache=kvc,
            )
            new_kv.append(kv)
            x = h
        return x, new_kv


class ThoughtAdapter(nn.Module):
    """Rescale reasoning hidden states to match item-embedding norm magnitude."""

    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    @torch.no_grad()
    def _target_norm(self, emb_weight):
        return emb_weight.norm(dim=-1).mean()

    def forward(self, h, emb_weight):
        tn = self._target_norm(emb_weight).detach()
        hn = h.norm(dim=-1, keepdim=True)
        return h * (tn / hn) * self.scale
