import torch
import torch.nn as nn
import math


# =========================================================
# 1. YaRN Config
# =========================================================
class YaRNConfig:
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_position: int = 4096,
        original_max_position: int = 2048,
        scale_factor: float = 8.0,
        attn_factor: float = 1.0,
        beta_fast: float = 32,
        beta_slow: float = 1,
    ):
        self.dim = dim
        self.base = base

        self.max_position = max_position
        self.original_max_position = original_max_position

        self.scale_factor = scale_factor
        self.attn_factor = attn_factor

        self.beta_fast = beta_fast
        self.beta_slow = beta_slow


# =========================================================
# 2. NTK-by-parts frequency scaling (core idea，模块1)
# =========================================================
def build_ntk_scaled_freqs(cfg: YaRNConfig, device=None):
    """
    returns inv_freq: [dim/2]
    """

    half_dim = cfg.dim // 2

    # base frequencies
    inv_freq = 1.0 / (
        cfg.base ** (torch.arange(0, half_dim, device=device).float() / half_dim)
    )

    # position scaling factor (YaRN core trick)
    factor = cfg.scale_factor

    # NTK-aware correction
    # intuition: preserve low freq, stretch high freq
    # 定出两个阈值，划分低中高频
    low_mask = inv_freq <= (1.0 / cfg.beta_fast)
    high_mask = inv_freq >= (1.0 / cfg.beta_slow)

    inv_freq = torch.where( # 这里其实相当于一个逐元素的if: {xx} else: {if: xxx else: xxx}
        low_mask,
        inv_freq,
        torch.where(
            high_mask,
            inv_freq / factor,
            inv_freq / math.sqrt(factor),
        ),
    )

    return inv_freq


# =========================================================
# 3. YaRN RoPE Module (industrial version)
# =========================================================
class YaRNRoPE(nn.Module):
    def __init__(self, cfg: YaRNConfig):
        super().__init__()
        self.cfg = cfg

        self.inv_freq = build_ntk_scaled_freqs(cfg) # 模块1修改了RoPE的固定频率

        # cache
        self._cos_cache = None
        self._sin_cache = None
        self._seq_len_cached = 0

    def _build_cache(self, seq_len: int, device): # cos和sin的缓存。因为这玩意和x无关，inference阶段多次forward的时候，缓存下来可以加速。
        if self._seq_len_cached >= seq_len:
            return

        pos = torch.arange(seq_len, device=device).float()  # [T]

        # 这里的"i,j->ij"是计算外积的语法糖，这句等效于freqs = pos[:,None] * self.inv_freq[None,:]
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)  # [T, D/2]

        # standard RoPE
        self._cos_cache = freqs.cos()
        self._sin_cache = freqs.sin()

        self._seq_len_cached = seq_len

    def forward(self, x, seq_dim=1):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        device = x.device

        self._build_cache(T, device)

        cos = self._cos_cache[:T][None, :, :]  # [1, T, D/2]
        sin = self._sin_cache[:T][None, :, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        out = torch.empty_like(x)

        # RoPE rotation
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos

        return out

# =========================================================
# 4. Attention Temperature Scaling (模块2)
# =========================================================
class YaRNAttentionScaling(nn.Module):
    def __init__(self, attn_factor: float = 1.0):
        super().__init__()
        self.attn_factor = attn_factor

    def forward(self, attn_logits):
        # scale logits before softmax
        return attn_logits * self.attn_factor

# =========================================================
# 5. Attention Calculating
# =========================================================
def apply_attention(q, k, rope: YaRNRoPE, scaler: YaRNAttentionScaling):
    """
    q, k: [B, T, D]
    """

    B, T, D = q.shape

    q = rope(q)
    k = rope(k)

    # standard attention logits (QK^T/√d)
    attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)

    # YaRN temperature correction
    attn_logits = scaler(attn_logits)

    return attn_logits