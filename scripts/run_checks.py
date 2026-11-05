"""Run the open check battery against the implementation in ``src``.

Usage:
    python scripts/run_checks.py

Exit code 0 means all checks passed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import minigpt
from src.checks import run


def main() -> int:
    implementation = SimpleNamespace(
        GPTConfig=minigpt.GPTConfig,
        sinusoidal_positional_encoding=minigpt.sinusoidal_positional_encoding,
        LayerNorm=minigpt.LayerNorm,
        split_heads=minigpt.split_heads,
        merge_heads=minigpt.merge_heads,
        scaled_dot_product_attention=minigpt.scaled_dot_product_attention,
        MultiHeadAttention=minigpt.MultiHeadAttention,
        FeedForward=minigpt.FeedForward,
        TransformerBlock=minigpt.TransformerBlock,
        MiniGPT=minigpt.MiniGPT,
        generate=minigpt.generate,
    )
    failures = run(implementation)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
