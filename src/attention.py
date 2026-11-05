"""Реэкспорт четырёх функций шага 1 из полной реализации.

Полный код лежит в :mod:`src.minigpt`; этот модуль нужен только для явной
точки входа к заданию «Внимание своими руками».
"""
from .minigpt import (  # noqa: F401
    TransformerBlock,
    merge_heads,
    scaled_dot_product_attention,
    split_heads,
)

__all__ = [
    "split_heads",
    "merge_heads",
    "scaled_dot_product_attention",
    "TransformerBlock",
]
