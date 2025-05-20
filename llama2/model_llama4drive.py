# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math, os
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union, List

import torch
import torch.distributions as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings, ModelOutput
from transformers.models.llama.configuration_llama import LlamaConfig

from accelerate.hooks import remove_hook_from_submodules
# from peft.config import PeftConfig
from safetensors.torch import save_file as safe_save_file
from peft.peft_model import PeftModelForCausalLM
from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    get_peft_model_state_dict,
)

from gameformer.predictor import Encoder as GameformerEncoder
from gameformer.predictor import LLMEnhancedGameFormer
from gameformer.predictor_adapter import LLMEnhancedGameFormer_Adapter
from gameformer.predictor_modules import CrossTransformer
from gameformer.train_utils import *
from llama2.trt_infer_singleton import TRTInferSingleton
from llama2.trt_infer_singleton import ONNXInferSingleton

from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

@dataclass
class BaseModelOutputWithPastDrive(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loca: Optional[List[Tuple[int]]] = None


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


@dataclass
class CausalLMOutputWithPastWithModel(CausalLMOutputWithPast):
    """
    Base class for causal language model (or autoregressive) outputs with label return.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Ground truth target sequence (masked) tokens.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    llm_loss: Optional[torch.FloatTensor] = None
    llm_regression_loss: Optional[torch.FloatTensor] = None
    llm_multi_head_loss: Optional[torch.FloatTensor] = None
    urban_loss: Optional[torch.FloatTensor] = None
    v_a_loss: Optional[torch.FloatTensor] = None
    neighbour_lane_loss: Optional[torch.FloatTensor] = None
    acc_class_loss: Optional[torch.FloatTensor] = None
    lane_change_loss: Optional[torch.FloatTensor] = None
    traffic_light_loss: Optional[torch.FloatTensor] = None
    gameformer_loss: Optional[torch.FloatTensor] = None
    gmm_loss: Optional[torch.FloatTensor] = None
    gameformer_planner_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    predictions: Optional[Tuple[torch.FloatTensor]] = None
    plan: Optional[Tuple[torch.FloatTensor]] = None
    llm_plan: Optional[Tuple[torch.FloatTensor]] = None


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

        self.gradient_checkpointing = False
        self.special_token_id = config.special_token_dict['<map>']
        # Initialize weights and apply final processing
        self.post_init()
        self.onnx_model_path = config.onnx_model_path
        self.tensorrt_model_path = config.tensorrt_model_path
        self.inference_model_type = config.inference_model_type

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        map_feats: torch.FloatTensor = None,
        map_masks: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if past_key_values is not None:
            input_ids_clone = input_ids.clone()
            input_ids = input_ids[:, -1:]

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if map_feats is not None and past_key_values is None:
            new_tokens_num = input_ids.shape[1] + map_feats.shape[1]
            special_toks_mask = input_ids == self.special_token_id
            special_toks_loc = torch.where(special_toks_mask)[1]
            seq_length = seq_length + map_feats.shape[1]
            seq_length_with_past = seq_length_with_past + map_feats.shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        loca = []
        if map_feats is not None and past_key_values is None:
            new_inputs_embeds = torch.zeros((batch_size, new_tokens_num, inputs_embeds.shape[-1]), device=input_ids.device).to(inputs_embeds.dtype)
            new_inputs_attention_mask = torch.zeros((batch_size, new_tokens_num), dtype=torch.bool, device=input_ids.device).to(inputs_embeds.dtype)
            new_labels = torch.zeros((batch_size, new_tokens_num), dtype=torch.long, device=input_ids.device)
            
            position_ids_bs = position_ids.shape[0]
            new_position_ids = torch.zeros((position_ids_bs, new_tokens_num), dtype=torch.long, device=input_ids.device)
            
            for b in range(batch_size):
                return_special_toks_loc = special_toks_loc
                new_inputs_embeds[b, :special_toks_loc[b]+1] = inputs_embeds[b, :special_toks_loc[b]+1]
                new_inputs_embeds[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = map_feats[b]
                new_inputs_embeds[b, special_toks_loc[b]+map_feats.shape[1]+1:] = inputs_embeds[b, special_toks_loc[b]+1:]
                loca.append((special_toks_loc[b]+1, special_toks_loc[b]+map_feats.shape[1]+1))
                if labels is not None:
                    new_labels[b, :special_toks_loc[b]+1] = labels[b, :special_toks_loc[b]+1]
                    new_labels[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = -100
                    new_labels[b, special_toks_loc[b]+map_feats.shape[1]+1:] = labels[b, special_toks_loc[b]+1:]
                
                if b < position_ids_bs:
                    new_position_ids[b, :special_toks_loc[b]+1] = position_ids[b, :special_toks_loc[b]+1]
                    new_position_ids[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = torch.arange(position_ids[b, special_toks_loc[b]]+1, position_ids[b, special_toks_loc[b]]+map_feats.shape[1]+1)
                    new_position_ids[b, special_toks_loc[b]+map_feats.shape[1]+1:] = position_ids[b, special_toks_loc[b]+1:] + map_feats.shape[1]
                
                if attention_mask is not None:
                    new_inputs_attention_mask[b, :special_toks_loc[b]+1] = attention_mask[b, :special_toks_loc[b]+1]
                    new_inputs_attention_mask[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = ~map_masks[b]
                    new_inputs_attention_mask[b, special_toks_loc[b]+map_feats.shape[1]+1:] = attention_mask[b, special_toks_loc[b]+1:]
            
            inputs_embeds = new_inputs_embeds
            attention_mask = new_inputs_attention_mask
            labels = new_labels
            position_ids = new_position_ids

        if map_feats is not None and past_key_values is not None:
            special_toks_loc = torch.where(input_ids_clone == self.special_token_id)[1]
            new_inputs_attention_mask = torch.zeros((batch_size, seq_length_with_past), dtype=torch.bool, device=input_ids.device).to(inputs_embeds.dtype)
            for b in range(batch_size):
                new_inputs_attention_mask[b, :special_toks_loc[b]+1] = attention_mask[b, :special_toks_loc[b]+1]
                new_inputs_attention_mask[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = ~ map_masks[b]
                new_inputs_attention_mask[b, special_toks_loc[b]+map_feats.shape[1]+1:] = attention_mask[b, special_toks_loc[b]+1:]
            attention_mask = new_inputs_attention_mask
            position_ids += map_feats.shape[1]

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask_2d = attention_mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Decide which inference engine to use: 'torch', 'onnx', or 'tensorrt'
        inference_model_type = self.inference_model_type
        print("the inference_model_type is: ",self.inference_model_type)

        if inference_model_type == 'torch':
            for idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, past_key_value, output_attentions)
                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

        elif inference_model_type == 'onnx':
            infer_engine = ONNXInferSingleton(self.onnx_model_path, providers=['CUDAExecutionProvider'])
            hidden_states = infer_engine.infer_hidden_states(inputs_embeds, attention_mask_2d, position_ids)

        elif inference_model_type == 'tensorrt':
            infer_engine = TRTInferSingleton(engine_path=self.tensorrt_model_path, device_id=0)
            hidden_states = infer_engine.infer_hidden_states(inputs_embeds, attention_mask_2d, position_ids)

        else:
            raise ValueError(f"Unknown inference backend: {inference_model_type}")

        return_dict = return_dict if return_dict is not None else True
        if not return_dict:
            return hidden_states, None, None, None
        
        return BaseModelOutputWithPastDrive(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            loca=[(torch.tensor(0), torch.tensor(0))],
        ), labels, new_inputs_attention_mask, return_special_toks_loc


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _keep_in_fp32_modules = ['map_adapter',
                             'waypoints_fc',
                             'waypoints_predictor',
                             'waypoints_output',
                             'map_encoder',
                             'gameformer',
                             'ego_v_a_predictor',
                             'neighbour_lane',
                             'acc_classification',
                             'lane_change',
                             'traffic_light',
                             'feature_adpter']

    _keep_small_lr_modules = [
            'gameformer',
        ]
    adapter_name_list = _keep_in_fp32_modules

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.feature_len = config.feature_len # number of waypoint, default=80
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Add Map adapter layers
        self.map_insize = config.map_insize
        self.map_adapter = nn.Linear(self.map_insize, config.hidden_size, bias=False)

        self.waypoints_predictor = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, self.feature_len*2))

        # regression
        self.ego_v_a_predictor = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 4))
        # classification
        self.neighbour_lane = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 2),
                                                    nn.Sigmoid())
        self.acc_classification = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 3),
                                                    nn.Softmax(dim=-1))
        self.lane_change = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 1),
                                                    nn.Sigmoid())
        self.traffic_light = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 4),
                                                    nn.Softmax(dim=-1))
            
        self.use_all_tokens = config.use_all_tokens
        self.adapter_fusion = config.adapter_fusion
        self.llm_inf_step = config.llm_inf_step

        if self.adapter_fusion:
            self.gameformer =  LLMEnhancedGameFormer_Adapter(encoder_layers=3, decoder_levels=2, modalities=6, neighbors=10) # this
        else:
            self.gameformer =  LLMEnhancedGameFormer(encoder_layers=3, decoder_levels=2, modalities=6, neighbors=10)
        self.map_encoder = GameformerEncoder(layers=3)
        self.feature_adpter = nn.Linear(self.model.config.hidden_size, 256)
            
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        try:
            number_weight = config.number_weight
        except:
            number_weight = 1.0
        weighted_mask = torch.ones(config.vocab_size, dtype=torch.float32)
        if number_weight > 1:
            number_tokens = [
                448,
                29900,
                29889,
                29896,
                29906,
                29941,
                29946,
                29945,
                29953,
                29955,
                29947,
                29929,
            ]  # -0.123456789
            weighted_mask[number_tokens] = number_weight
        self.weighted_mask = weighted_mask
        # Initialize weights and apply final processing
        self.post_init()

    def reset_trainable_param(self):
        for name, param in self.named_parameters():
            if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in self._keep_in_fp32_modules):
                param.requires_grad = True
        
        if self.adapter_fusion:
            for name, param in self.gameformer.named_parameters():
                if 'llm_adapt_attention' in name:
                    param.requires_grad = True

        if self.config.enable_lora:
            for name, param in self.model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

    def reinit_weights(self):
        init_module_list = [name for name in self.adapter_name_list if hasattr(self, name)]
        for name in init_module_list:
            print(f"Reinit {name} weights")
            for module in getattr(self, name).modules():
                # if isinstance(module, nn.LSTM):
                #     import pdb; pdb.set_trace()
                if hasattr(module, '_reset_parameters'):
                    module._reset_parameters()
                elif hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                elif hasattr(module, 'flatten_parameters'):
                    module.flatten_parameters()
                # else:
                #     print(f"Module {module} has no reset_parameters or _reset_parameters method")
        
    def resume_from_checkpoint(self ,ckpt_dir, gameformer_ckpt=False):
        if gameformer_ckpt:
            weights = torch.load(ckpt_dir, map_location=torch.device('cpu'))

            self.gameformer.load_state_dict(weights, strict=False)

            processed_weights = OrderedDict()
            for key, value in weights.items():
                if key.startswith("encoder."):
                    new_key = key[len("encoder."):]
                    processed_weights[new_key] = value
            if len(processed_weights) == 0:
                processed_weights = weights
            self.map_encoder.load_state_dict(processed_weights, strict=False)
        # elif lora_ckpt:
        #     weights = torch.load(ckpt_dir)
        #     set_peft_model_state_dict(self, weights)
        #     print('LoRA pretrain weights have been loaded')
        else:
            if os.path.isdir(ckpt_dir):
                ckpt_ls = os.listdir(ckpt_dir)
                for ckpt in ckpt_ls:
                    if '.bin' in ckpt:
                        weights = torch.load(ckpt_dir+'/'+ckpt, map_location=torch.device('cpu'))
                        module = getattr(self, ckpt.split('.')[0], None)
                        if ckpt == 'embed_tokens.bin':
                            self.load_state_dict(weights, strict=False)
                        elif module is None:
                            print("%s could not be loaded successfully"%str(ckpt))
                        else:
                            try:
                                module.load_state_dict(weights, strict=True)
                                module.to(self.model.device)
                            except:
                                print("%s could not be loaded successfully"%str(ckpt))
            else:
                weights = torch.load(ckpt_dir)
                self.gameformer.load_state_dict(weights, strict=True)
        self.map_encoder.to(self.model.device)
        self.to(self.model.device)

    def reload_mapencoder_weights(self):
        self.reinit_weights()
        if self.config.mapEncoder_pretrain_weight is None:
            return
        pretrain_weights = torch.load(self.config.mapEncoder_pretrain_weight, map_location=torch.device('cpu'))
        self.gameformer.load_state_dict(pretrain_weights, strict=False)
        self.gameformer.to(self.model.device)
        processed_weights = OrderedDict()
        for key, value in pretrain_weights.items():
            if key.startswith("encoder."):
                new_key = key[len("encoder."):]
                processed_weights[new_key] = value
        if len(processed_weights) == 0:
            processed_weights = pretrain_weights
        self.map_encoder.load_state_dict(processed_weights, strict=False)
        self.map_encoder.to(self.model.device)

    def cuda(self, *args, **kwargs):
        return nn.Module.cuda(self, *args, **kwargs)

    def to(self, *args, **kwargs):
        return nn.Module.to(self, *args, **kwargs)

    def half(self, *args):
        return nn.Module.half(self)

    def float(self, *args):
        return nn.Module.float(self)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPastWithModel, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        map_feats: Optional[torch.FloatTensor] = None,
        map_masks: Optional[torch.FloatTensor] = None,
        urban_features = None,
        urban_avails = None,
        ego_agent_past: Optional[torch.FloatTensor] = None,
        neighbor_agents_past: Optional[torch.FloatTensor] = None,
        map_lanes: Optional[torch.FloatTensor] = None,
        map_crosswalks: Optional[torch.FloatTensor] = None,
        route_lanes: Optional[torch.FloatTensor] = None,
        ego_future: Optional[torch.FloatTensor] = None,
        neighbors_future: Optional[torch.FloatTensor] = None,
        cur_iter: Optional[torch.LongTensor] = 1,
        ego_v_a: Optional[torch.FloatTensor] = None,
        neighbour_lane: Optional[torch.FloatTensor] = None,
        acc_classification: Optional[torch.FloatTensor] = None,
        lane_change: Optional[torch.FloatTensor] = None,
        traffic_light: Optional[torch.FloatTensor] = None,
        ego_lane_flag: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inference = False,
    ) -> Union[Tuple, CausalLMOutputWithPastWithModel]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        
        # gt for gameformer
        # import ipdb; ipdb.set_trace()
        if not inference:
            ego_future_gt = ego_future
            neighbors_future_gt = neighbors_future
            neighbors_future_valid_gt = torch.ne(neighbors_future_gt[..., :2], 0)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if map_feats is not None:
            map_feats = map_feats.to(self.map_adapter.weight.dtype)
            map_feats = self.map_adapter(map_feats)
            map_feats = map_feats.to(self.map_adapter.weight.dtype)
        if ego_agent_past is not None:
            assert map_feats is None
            raw_map_vector = {
                'ego_agent_past': ego_agent_past.to(self.map_adapter.weight.dtype), #[1, 21, 7]
                'neighbor_agents_past': neighbor_agents_past.to(self.map_adapter.weight.dtype),
                'map_lanes': map_lanes.to(self.map_adapter.weight.dtype), # [16, 40, 50, 7]
                'map_crosswalks': map_crosswalks.to(self.map_adapter.weight.dtype),
                'route_lanes': route_lanes.to(self.map_adapter.weight.dtype), # [16, 10, 50, 3]
            }
            encoder_outputs = self.map_encoder(raw_map_vector)
            map_feats, map_masks = encoder_outputs['encoding'], encoder_outputs['mask']
            if torch.isnan(map_feats).any():
                import pdb; pdb.set_trace()
            map_feats = self.map_adapter(map_feats)
            map_feats = map_feats.to(self.map_adapter.weight.dtype)
        else:
            raise NotImplementedError()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        outputs, labels, new_inputs_attention_mask, feature_position = self.model(
            input_ids=input_ids,
            labels=labels,
            map_feats=map_feats,
            map_masks=map_masks,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ego_plan = None
        level_k_outputs = None
        hidden_states = outputs[0]

        if cur_iter.index % self.llm_inf_step != 0:
            hidden_states = self.prev_hidden_states
        else:
            self.prev_hidden_states = hidden_states
        
        # use query feature instead of direct hidden_states
        ########
        if self.use_all_tokens:
            pooling_features = hidden_states.mean(dim=1)
            hidden_states = pooling_features
            predicted_feature = self.feature_adpter(pooling_features)
        else:
            hidden_states = hidden_states[:, -1, :]
            predicted_feature = self.feature_adpter(hidden_states)
            
        # loss for llm hidden feature
        predicted_waypoints = self.waypoints_predictor(hidden_states)
        predicted_waypoints = predicted_waypoints.reshape(predicted_waypoints.shape[0], self.feature_len, 2)
        if not inference:
            waypoints_loss = F.smooth_l1_loss(predicted_waypoints, ego_future[..., :2])
            waypoints_loss += F.smooth_l1_loss(predicted_waypoints[:, -1], ego_future[:, -1, :2])

        predicted_ego_v_a = self.ego_v_a_predictor(hidden_states)
        predicted_ego_v_a = predicted_ego_v_a.reshape(predicted_ego_v_a.shape[0], 4)
        if not inference:
            v_a_loss = F.smooth_l1_loss(predicted_ego_v_a, ego_v_a)
        
        predicted_neighbour_lane = self.neighbour_lane(hidden_states)
        predicted_neighbour_lane = predicted_neighbour_lane.reshape(predicted_neighbour_lane.shape[0], 2)
        if not inference:
            neighbour_lane_loss = F.binary_cross_entropy(predicted_neighbour_lane, torch.tensor(neighbour_lane.squeeze(-1), dtype=torch.float32))
        
        pred_acc_classification = self.acc_classification(hidden_states)
        pred_acc_classification = pred_acc_classification.reshape(pred_acc_classification.shape[0], 3)
        if not inference:
            acc_class_loss = F.cross_entropy(pred_acc_classification, torch.tensor(acc_classification, dtype=torch.float32))
        
        pred_lane_change = self.lane_change(hidden_states)
        pred_lane_change = pred_lane_change.reshape(pred_lane_change.shape[0], 1)
        if not inference:
            lane_change_loss = F.binary_cross_entropy(pred_lane_change, torch.tensor(lane_change, dtype=torch.float32))
        
        pred_traffic_light = self.traffic_light(hidden_states)
        pred_traffic_light = pred_traffic_light.reshape(pred_traffic_light.shape[0], 4)
        if not inference:
            traffic_light_loss = F.cross_entropy(pred_traffic_light, torch.tensor(traffic_light, dtype=torch.float32))
        
        if not inference:
            llm_multi_head_loss = v_a_loss + neighbour_lane_loss + acc_class_loss + lane_change_loss + traffic_light_loss
            llm_loss = 0.5*waypoints_loss + 0.5*llm_multi_head_loss
        else:
            llm_loss = None
            waypoints_loss = None
            llm_multi_head_loss = None
            v_a_loss = None
            neighbour_lane_loss = None
            acc_class_loss = None
            lane_change_loss = None
            traffic_light_loss = None
            gameformer_loss = None
            gmm_loss = None
            plan_loss = None
        
        if len(predicted_feature.shape)<3:
            llm_feature = predicted_feature.unsqueeze(1)
        else:
            llm_feature = predicted_feature

        input_t = (raw_map_vector, llm_feature)
        level_k_outputs, ego_plan = self.gameformer(input_t)
        
        if not inference:
            gmm_loss, results = level_k_loss(level_k_outputs, ego_future_gt[..., :2], neighbors_future_gt[:,:self.gameformer.neighbors,:,:2], neighbors_future_valid_gt[:,:self.gameformer.neighbors,...])
            gmm_loss = torch.abs(gmm_loss)
            plan_loss = planning_loss(ego_plan, ego_future_gt[..., :2])
            gameformer_loss = gmm_loss + plan_loss
            loss = gameformer_loss + llm_loss
        
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastWithModel(
            loss=loss if not inference else None,
            llm_loss=llm_loss,
            llm_regression_loss=waypoints_loss,
            llm_multi_head_loss=llm_multi_head_loss,
            urban_loss=None,
            v_a_loss=v_a_loss,
            neighbour_lane_loss=neighbour_lane_loss,
            acc_class_loss=acc_class_loss,
            lane_change_loss=lane_change_loss,
            traffic_light_loss=traffic_light_loss,
            gameformer_loss=gameformer_loss,
            gmm_loss=gmm_loss,
            gameformer_planner_loss=plan_loss,
            logits=logits,
            labels=labels,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            predictions = level_k_outputs,
            plan = ego_plan,
            llm_plan = predicted_waypoints,
        )

    def save_pretrained(
            self,
            save_directory: str,
            safe_serialization: bool = False,
            selected_adapters: Optional[List[str]] = None,
            **kwargs: Any,
    ):
        r"""
        This function saves the map adapteer, along with the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        # save map adapter weight
        model_names = kwargs.get("state_dict", None).keys()
        for name in self.adapter_name_list:
            module = getattr(self, name, None)
            if module and any(param.requires_grad for param in module.parameters()):
                if safe_serialization:
                    safe_save_file(
                        module.state_dict(),
                        os.path.join(save_directory, f'{name}.safetensors'),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(module.state_dict(), os.path.join(save_directory, f'{name}.bin'))
                print(f"Save {name}")

        # save global dict
        embed_tokens_name = None
        for model_name in model_names:
            if 'embed_tokens' in model_name:
                embed_tokens_name = model_name
                break

        os.makedirs(save_directory, exist_ok=True)

        if embed_tokens_name is None:
            raise ValueError(f"Cannot find embed_tokens in the model state dict.")
        else:
            embed_tokens_weight_dict = dict()
            embed_tokens_weight_dict[embed_tokens_name] = kwargs.get("state_dict", None)[embed_tokens_name]
            if safe_serialization:
                safe_save_file(
                    embed_tokens_weight_dict,
                    os.path.join(save_directory, 'embed_tokens.safetensors'),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(embed_tokens_weight_dict, os.path.join(save_directory, 'embed_tokens.bin'))

    def load_weights(self, model_id):
        if model_id is None:
            print('!!!!  No model id, not loaded at all')
            return
        if self.config.mapEncoder_pretrain_weight is None:
            self.config.mapEncoder_pretrain_weight = os.path.join(model_id, f'map_encoder.bin')
        self.reload_mapencoder_weights()
        for map_encoder_name in self.adapter_name_list:
            # if 'gameformer' in map_encoder_name:
            #     import pdb; pdb.set_trace()
            try:
                loaded_weight = torch.load(os.path.join(model_id, f'{map_encoder_name}.bin'))
                new_weight = OrderedDict()
                for k in loaded_weight:
                    new_weight[f'{map_encoder_name}.{k}'] = loaded_weight[k]
                loaded_weight = new_weight
            except:
                print(f'Error in load {map_encoder_name}')
                continue
            # print(f'Success in load {map_encoder_name}')

            for name, param in self.named_parameters():
                if name in loaded_weight.keys():
                    # if 'motion' in name:
                    #     import pdb; pdb.set_trace()
                    param.data.copy_(loaded_weight[name])
                    del loaded_weight[name]
                    print(f"Load {map_encoder_name} weight {name} from {model_id}")
            
            if len(loaded_weight.keys())!=0:
                for k in loaded_weight.keys():
                    print('%s has not been successfully loaded!!!!!!!!!!!!!'%str(k))

        try:
            loaded_weight = torch.load(os.path.join(model_id, 'embed_tokens.bin'))
        except:
            print(' error in load embed tokens')
            loaded_weight = {}
        for name, param in self.named_parameters():
            if name in loaded_weight.keys():
                param.data.copy_(loaded_weight[name])
                # param.requires_grad = is_trainable
                print(f"Load embed_tokens weight {name} from {model_id}")
                

    def prepare_inputs_for_generation(
        self, input_ids, map_feats, map_masks, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "map_feats": map_feats,
                "map_masks": map_masks,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = model_embeds.weight.shape[0]
        self.vocab_size = model_embeds.weight.shape[0]
        
        # Resize label weight
        if hasattr(self, "weighted_mask"):
            try:
                number_weight = self.config.number_weight
            except:
                number_weight = 1.0
            weighted_mask = torch.ones(self.config.vocab_size, dtype=torch.float32)
            if number_weight > 1:
                number_tokens = [
                    448,
                    29900,
                    29889,
                    29896,
                    29906,
                    29941,
                    29946,
                    29945,
                    29953,
                    29955,
                    29947,
                    29929,
                ]  # -0.123456789
                weighted_mask[number_tokens] = number_weight
            self.weighted_mask = weighted_mask

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds
    


class ModelWithLoRA(PeftModelForCausalLM):
    def __init__(self, model, peft_config, num_vector_tokens=64):
        super().__init__(model, peft_config)
        self.num_vector_tokens = num_vector_tokens
        self.to(model.device)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPastWithModel]:

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        # loss = outputs.loss

        # return {"loss": loss}
        return outputs
    
    def resume_lora_from_checkpoint(self, ckpt_path):
        weights = torch.load(ckpt_path)
        set_peft_model_state_dict(self, weights)
        print('LoRA pretrain weights have been loaded')


    def load_weights(self, model_id):
        if model_id is None:
            print('!!!!  No model id, not loaded at all')
            return
        lora_ckpt = os.path.join(model_id, "adapter_model.bin")
        lora_weights = torch.load(lora_ckpt)
        set_peft_model_state_dict(self, lora_weights)
        print('LoRA weights have been loaded')

        global_ckpt_ = [dir_path for dir_path in os.listdir(model_id) if 'global' in dir_path][0]
        global_ckpt_ = os.path.join(model_id, global_ckpt_)
        model_states = [dir_path for dir_path in os.listdir(global_ckpt_) if 'model' in dir_path][0]
        model_ckpt_ = os.path.join(global_ckpt_, model_states)
        model_weights = torch.load(model_ckpt_)['module']
        for name, param in self.named_parameters():
                if name in model_weights.keys():
                    param.data.copy_(model_weights[name])
                    del model_weights[name]
                    print(f"Load {name} weight from {model_ckpt_}")

        if len(model_weights.keys())!=0:
                for k in model_weights.keys():
                    print('%s has not been successfully loaded!!!!!!!!!!!!!'%str(k))
        self.to(self.model.device)


        
