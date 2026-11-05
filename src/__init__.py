"""Пакет с реализацией и проверками ДЗ1."""

from .checks import CHECKS, run
from .minigpt import (
    FeedForward,
    GPTConfig,
    LayerNorm,
    MiniGPT,
    MultiHeadAttention,
    TransformerBlock,
    generate,
    merge_heads,
    scaled_dot_product_attention,
    sinusoidal_positional_encoding,
    split_heads,
)

__all__ = [
    "CHECKS",
    "run",
    "GPTConfig",
    "MiniGPT",
    "LayerNorm",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "generate",
    "split_heads",
    "merge_heads",
    "scaled_dot_product_attention",
    "sinusoidal_positional_encoding",
]
