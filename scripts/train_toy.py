"""Reproduce the short training run and the generation example.

Usage:
    python scripts/train_toy.py
    python scripts/train_toy.py --steps 600 --seed 0 --out results

Writes ``loss_curve.png``, ``generated_text.txt`` and ``training_metrics.json``
into the output directory (``results`` by default).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minigpt import GPTConfig, MiniGPT, generate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600, help="optimisation steps (default: 600)")
    parser.add_argument("--seed", type=int, default=0, help="torch seed (default: 0)")
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="output directory")
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    text = "алгоритм внимания смотрит на все позиции сразу. " * 200
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text])

    cfg = GPTConfig(
        vocab_size=len(vocab),
        context_size=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    )
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Held-out tail: never used for weight updates.
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]

    def batch(size: int = 32, source: torch.Tensor = train_data):
        pos = torch.randint(0, len(source) - cfg.context_size - 1, (size,))
        x = torch.stack([source[p:p + cfg.context_size] for p in pos])
        y = torch.stack([source[p + 1:p + cfg.context_size + 1] for p in pos])
        return x, y

    # Fixed validation windows keep the validation curve comparable across steps.
    val_pos = torch.linspace(0, len(val_data) - cfg.context_size - 2, 8).long()
    val_x = torch.stack([val_data[p:p + cfg.context_size] for p in val_pos])
    val_y = torch.stack([val_data[p + 1:p + cfg.context_size + 1] for p in val_pos])

    @torch.no_grad()
    def validation_loss() -> float:
        model.eval()
        _, loss, _ = model(val_x, targets=val_y)
        model.train()
        return float(loss.item())

    history: list[float] = []
    val_steps: list[int] = []
    val_history: list[float] = []
    started = time.perf_counter()

    for step in range(args.steps):
        x, y = batch()
        _, loss, _ = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if step % 50 == 0 or step == args.steps - 1:
            val_steps.append(step)
            val_history.append(validation_loss())
        if step % 100 == 0:
            print(f"шаг {step:4d}  потери {loss.item():.4f}  val {val_history[-1]:.4f}")

    elapsed = time.perf_counter() - started
    print(
        f"итог: {history[0]:.3f} -> {history[-1]:.3f}; "
        f"val {val_history[0]:.3f} -> {val_history[-1]:.3f}"
    )
    print(f"время обучения: {elapsed:.1f} с")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(history, label="обучение")
    ax.plot(val_steps, val_history, "o-", label="валидация")
    ax.set_xlabel("шаг обучения")
    ax.set_ylabel("потери")
    ax.set_title("Кривая потерь после исправления багов")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=150)
    plt.close(fig)

    start = torch.tensor([[stoi["а"]]])
    generated_ids = generate(model, start, max_new_tokens=200, temperature=0.8, top_k=10)
    generated = "".join(itos[int(i)] for i in generated_ids[0])
    (out / "generated_text.txt").write_text(generated, encoding="utf-8")

    metrics = {
        "seed": args.seed,
        "steps": args.steps,
        "elapsed_sec": round(elapsed, 2),
        "train_loss_first": history[0],
        "train_loss_last": history[-1],
        "val_loss_first": val_history[0],
        "val_loss_last": val_history[-1],
        "config": {
            "context_size": cfg.context_size,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "dropout": cfg.dropout,
            "batch_size": 32,
        },
    }
    (out / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"график: {out / 'loss_curve.png'}")
    print(f"генерация: {out / 'generated_text.txt'}")
    print(f"метрики: {out / 'training_metrics.json'}")
    print("\nПример сгенерированного текста:")
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
