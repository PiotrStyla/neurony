from transformers import PretrainedConfig


class GollemV6Config(PretrainedConfig):
    """GoLLeM-v5 (Fabryka AI / SlayerLab): Qwen3-style decoder with value residuals.

    Mirrors the published config.json fields (block/ffn/ffn_mult/n_embd/n_head/n_layer/
    norm/norm_eps/pos/qk_norm/rope_theta/tied_weights/value_residual/vocab) under
    standard transformers names. Semantics are defined by train_gpt_ref.py in
    SlayerLab/gollem-v5-ckpts.
    """

    model_type = "gollem_v6"

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 960,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 15,
        num_key_value_heads: int = 15,
        head_dim: int = 64,
        intermediate_size: int = 2560,
        ffn_mult: float = 2.667,
        logit_cap: float = 0.0,
        max_position_embeddings: int = 1024,
        rope_theta: float = 100000.0,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
        value_residual: bool = True,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.ffn_mult = ffn_mult
        self.logit_cap = logit_cap
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm = qk_norm
        self.value_residual = value_residual
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range
