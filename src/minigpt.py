"""MiniGPT: полная реализация из `notebooks/dz1-bughunt.ipynb`.

Версия с исправленными семантическими ошибками: позиционное кодирование
начинается с нуля, LayerNorm нормирует признаки внутри токена, softmax во
внимании берётся по ключам.

Файл сгенерирован из исполненного ноутбука, поэтому совпадает с кодом,
который проходит 17 проверок.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256
    context_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int | None = None      # по умолчанию 4 * d_model
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_heads:
            raise ValueError('d_model должно делиться на n_heads без остатка')

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# --- позиционное кодирование --------------------------------------------------

def sinusoidal_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    """Синусоидальное кодирование из «Attention is All You Need», форма (length, d_model).

    Чётные каналы — синус, нечётные — косинус; частота падает с номером канала.
    """
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    channel = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(-math.log(10000.0) * channel / d_model)
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# --- нормализация -------------------------------------------------------------

class LayerNorm(nn.Module):
    """Нормализация по признакам внутри токена (последняя ось), а не по батчу."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normalized + self.bias


# --- внимание -----------------------------------------------------------------

def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, L, d_model) -> (B, n_heads, L, d_head).

    Сначала разворачиваем последнюю ось в (n_heads, d_head), только потом переставляем
    оси. Обратный порядок перемешает каналы между головами.
    """
    batch, length, d_model = x.shape
    d_head = d_model // n_heads
    x = x.view(batch, length, n_heads, d_head)
    return x.transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, n_heads, L, d_head) -> (B, L, d_model). Обратная операция к split_heads."""
    batch, n_heads, length, d_head = x.shape
    x = x.transpose(1, 2).contiguous()
    return x.view(batch, length, n_heads * d_head)


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor,
                                 value: torch.Tensor,
                                 mask: torch.Tensor | None = None,
                                 dropout: nn.Module | None = None):
    """Внимание: softmax(QKᵀ / sqrt(d_k)) V. Возвращает (выход, веса внимания).

    Деление на корень из d_k держит дисперсию логитов независимой от размерности:
    без него softmax насыщается и градиенты гаснут. Маску применяем ДО softmax,
    подставляя -inf, чтобы запрещённые позиции получили нулевой вес.
    """
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ value, weights


class MultiHeadAttention(nn.Module):
    """Многоголовое причинное внимание."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.w_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        causal = torch.tril(torch.ones(cfg.context_size, cfg.context_size))
        self.register_buffer('causal_mask', causal.view(1, 1, cfg.context_size, cfg.context_size))

    def forward(self, x: torch.Tensor):
        length = x.size(1)
        q = split_heads(self.w_q(x), self.n_heads)
        k = split_heads(self.w_k(x), self.n_heads)
        v = split_heads(self.w_v(x), self.n_heads)
        mask = self.causal_mask[:, :, :length, :length]
        out, weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        return self.w_o(merge_heads(out)), weights


# --- полносвязный подслой -----------------------------------------------------

class FeedForward(nn.Module):
    """Две проекции с нелинейностью между ними.

    Без нелинейности композиция двух Linear схлопывается в один линейный слой,
    и подслой перестаёт что-либо добавлять.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.up(x))
        return self.dropout(self.down(hidden))


# --- блок и модель ------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm блок: x + attn(norm(x)), затем x + ffn(norm(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor):
        attn_out, weights = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, weights


class MiniGPT(nn.Module):
    """Декодер-трансформер: эмбеддинги, позиционное кодирование, блоки, голова."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer('pos_emb',
                             sinusoidal_positional_encoding(cfg.context_size, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_out = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        length = idx.size(1)
        if length > self.cfg.context_size:
            raise ValueError(f'последовательность {length} длиннее контекста {self.cfg.context_size}')
        x = self.token_emb(idx) + self.pos_emb[:length]
        x = self.drop(x)
        attentions = []
        for block in self.blocks:
            x, weights = block(x)
            attentions.append(weights)
        logits = self.lm_head(self.norm_out(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, attentions


@torch.no_grad()
def generate(model: MiniGPT, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """Авторегрессионная генерация. model.eval() обязателен: иначе работает dropout."""
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        window = idx[:, -model.cfg.context_size:]
        logits, _, _ = model(window)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    if was_training:
        model.train()
    return idx