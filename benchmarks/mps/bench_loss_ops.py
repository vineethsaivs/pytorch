"""
Benchmark for MPS BCE (binary cross-entropy) loss op.
Uses torch.utils.benchmark.Timer.blocked_autorange — no hand-rolled timing.

For reduction=none, fwd+bwd uses .backward(rand_weights) to model the real
use case (per-sample importance weighting). reduction=mean/sum use .sum().backward().

Covers:
  - binary_cross_entropy forward + backward
  - Standard shapes + LLM training workloads (LLaMA / GPT-4 scale)
  - float32, float16, bfloat16

Run: python benchmarks/mps/bench_loss_ops.py
"""

import itertools

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


DEVICE = "mps"
MIN_RUN = 1.0  # seconds per config for blocked_autorange

# Warm up the MPS device + PSO cache
for _ in range(30):
    torch.randn(1024, device=DEVICE).sum()
torch.mps.synchronize()

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCTIONS = ["none", "mean", "sum"]

# Pointwise (BCE) shapes.
# Includes standard small/medium/large + LLM-scale hidden-state tensors.
POINTWISE_SHAPES = [
    (4096,),  # 1D medium
    (1 << 20,),  # 1D large (1M)
    (32, 4096),  # batch=32, hidden=4096 (LLaMA hidden)
    (16, 512, 4096),  # batch=16, seq=512, hidden=4096 (LLM activations)
    (8, 2048, 4096),  # batch=8, seq=2048, hidden=4096
]


# ── helpers ────────────────────────────────────────────────────────────────


def hdr(title):
    print(f"\n{'─' * 100}")
    print(f"  {title}")
    print(f"{'─' * 100}")
    print(
        f"  {'shape':<28} {'dtype':<10} {'red':<6} {'fwd median µs':>15} {'fwd+bwd median µs':>19}"
    )
    print(f"  {'─' * 28} {'─' * 10} {'─' * 6} {'─' * 15} {'─' * 19}")


def row(shape, dtype, red, fwd, fwdbwd):
    dt = str(dtype).split(".")[-1]
    bwd_s = f"{fwdbwd:>19.2f}" if fwdbwd is not None else f"{'—':>19}"
    print(f"  {str(shape):<28} {dt:<10} {red:<6} {fwd:>15.2f} {bwd_s}")


def time_fwd(stmt, g):
    try:
        m = Timer(stmt=stmt, globals=g).blocked_autorange(min_run_time=MIN_RUN)
        return m.median * 1e6
    except Exception:
        return None


def time_fwdbwd(stmt, g):
    try:
        m = Timer(stmt=stmt, globals=g).blocked_autorange(min_run_time=MIN_RUN)
        return m.median * 1e6
    except Exception:
        return None


# ── BCE Loss ──────────────────────────────────────────────────────────────

hdr("BCELoss — forward   |   forward+backward")
for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
    try:
        xb = torch.sigmoid(torch.randn(shape, device=DEVICE, dtype=dtype))
        yb = torch.randint(0, 2, shape, device=DEVICE).to(dtype)
    except Exception:
        continue
    g = dict(
        F=F,
        x=xb,
        y=yb,
        red=red,
        xg=xb.detach().requires_grad_(True),
        w=torch.rand_like(xb),
    )
    fwd = time_fwd("F.binary_cross_entropy(x, y, reduction=red)", g)
    if fwd is None:
        continue
    bwd_stmt = (
        "xg.grad=None; F.binary_cross_entropy(xg, y, reduction='none').backward(w)"
        if red == "none"
        else "xg.grad=None; F.binary_cross_entropy(xg, y, reduction=red).sum().backward()"
    )
    fwdbwd = time_fwdbwd(bwd_stmt, g)
    row(shape, dtype, red, fwd, fwdbwd)


print()
