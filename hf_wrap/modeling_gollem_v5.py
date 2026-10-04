"""GoLLeM-v5 as a HuggingFace model.

Semantics are exactly ``train_gpt_ref.py`` (SlayerLab/gollem-v5-ckpts):
pre-RMSNorm blocks, fused ``qkv`` (bias) -> per-head QK-RMSNorm -> interleaved
RoPE -> value residual ``v += lambda_layer0 * v0`` (ResFormer) -> causal
attention -> ``o_proj`` (bias), SwiGLU MLP (no bias), tied embeddings, final
norm before the head.

Module names follow the Llama/Qwen3 tree (``model.layers.*.self_attn/mlp``,
``lm_head``) so structural discovery (interp-engine) finds decoder layers,
attention, MLP and the unembedding without a table entry. The shipped
``model.safetensors`` is the published checkpoint remapped onto these names
(see convert_weights.py; parity-checked against train_gpt_ref).

Attention goes through transformers' attention-interface dispatch so that
circuit-tracer's frozen-pattern implementation can be selected via
``config._attn_implementation``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_gollem_v5 import GollemV5Config


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Plain softmax attention, the fallback for an unknown implementation name."""
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    # transformers' attention-interface convention: return [B, T, H, D]
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GollemV5RMSNorm(nn.Module):
    """Qwen3-style RMSNorm with fp32 accumulation (identical to train_gpt_ref)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def apply_rope(x: torch.Tensor, base: float) -> torch.Tensor:
    """Interleaved-convention RoPE on [B, H, T, D] (train_gpt_ref parity)."""
    _, _, t, dim = x.shape
    pos = torch.arange(t, device=x.device, dtype=torch.float32)
    freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    ang = torch.outer(pos, freq)
    cos, sin = ang.cos().to(x.dtype)[None, None], ang.sin().to(x.dtype)[None, None]
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class GollemV5MLP(nn.Module):
    def __init__(self, config: GollemV5Config):
        super().__init__()
        hidden = int(round(config.ffn_mult * config.hidden_size))
        self.gate_proj = nn.Linear(config.hidden_size, hidden, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GollemV5Attention(nn.Module):
    def __init__(self, config: GollemV5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        if config.qk_norm:
            self.q_norm = GollemV5RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = GollemV5RMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = nn.Identity()
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # ResFormer value residual: every block except the first learns how much of
        # block 0's value to mix in (lambda init 0). None marks the first block,
        # which defines v0.
        self.vr_lambda = nn.Parameter(torch.zeros(1)) if config.value_residual and layer_idx > 0 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        v0: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        b, t, _ = hidden_states.shape
        q, k, v = self.qkv(hidden_states).split(self.hidden_size, dim=-1)
        shape = (b, t, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, self.config.rope_theta)
        k = apply_rope(k, self.config.rope_theta)
        if self.vr_lambda is not None:
            v = v + self.vr_lambda * v0
        new_v0 = v if self.vr_lambda is None else v0

        # subscript form kept verbatim: transformers' _can_set_attn_implementation() detects
        # AttentionInterface support by searching the forward source for this exact expression
        if self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        else:
            attention_interface = eager_attention_forward
        attn_output, attn_weights = attention_interface(
            self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling, **kwargs
        )
        attn_output = attn_output.reshape(b, t, self.hidden_size).contiguous()
        return self.o_proj(attn_output), attn_weights, new_v0


class GollemV5DecoderLayer(nn.Module):
    def __init__(self, config: GollemV5Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = GollemV5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GollemV5Attention(config, layer_idx)
        self.post_attention_layernorm = GollemV5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GollemV5MLP(config)

    def forward(self, hidden_states, v0, attention_mask=None, **kwargs):
        attn_out, attn_weights, v0 = self.self_attn(
            self.input_layernorm(hidden_states), v0, attention_mask=attention_mask, **kwargs
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, v0, attn_weights


def _make_causal_mask(batch: int, q_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    q_pos = torch.arange(q_len, device=device)[:, None]
    k_pos = torch.arange(q_len, device=device)[None, :]
    mask = torch.where(q_pos >= k_pos, torch.zeros((), device=device), torch.full((), min_val, device=device))
    return mask.to(dtype)[None, None].expand(batch, 1, q_len, q_len)


class GollemV5PreTrainedModel(PreTrainedModel):
    config_class = GollemV5Config
    base_model_prefix = "model"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


class GollemV5Model(GollemV5PreTrainedModel):
    def __init__(self, config: GollemV5Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            GollemV5DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = GollemV5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        b, t, _ = inputs_embeds.shape
        causal = _make_causal_mask(b, t, inputs_embeds.dtype, inputs_embeds.device)
        if attention_mask is not None:
            pad = (1 - attention_mask[:, None, None, :].to(inputs_embeds.dtype)) * torch.finfo(inputs_embeds.dtype).min
            causal = causal + pad
        hidden_states = inputs_embeds
        v0 = None
        for layer in self.layers:
            hidden_states, v0, _ = layer(hidden_states, v0, attention_mask=causal)
        # HF base-model output contract (interp-engine's arch.trunk reads .last_hidden_state)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states))


class GollemV5ForCausalLM(GollemV5PreTrainedModel, GenerationMixin):
    def __init__(self, config: GollemV5Config):
        super().__init__(config)
        self.model = GollemV5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        hidden_states = self.model(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds
        ).last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1)
            )
        if not return_dict:
            return ((logits,) if loss is None else (loss, logits))
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **{k: v for k, v in kwargs.items() if k in ("attention_mask", "past_key_values")}}
