"""MPS cross_entropy: fused Metal kernel vs the reference decomposed path.

Same-build A/B. Mode A = reference log_softmax + nll_loss (forced via
PYTORCH_MPS_FORCE_DECOMPOSED_CROSS_ENTROPY=1); mode B = the fused Metal CE kernel
(env unset). The toggle is read once per process (static in
can_use_fused_mps_cross_entropy), so each MODE runs in its own process:
  PYTORCH_MPS_FORCE_DECOMPOSED_CROSS_ENTROPY=1 BENCH_MODE=A python bench_crossentropy.py
  BENCH_MODE=B python bench_crossentropy.py
Methodology: amortized many-iter / one-sync (dodges the MPS sync floor).
Requires this checkout importable as torch (pip install -e . or PYTHONPATH).
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS not available")
DEV = torch.device("mps")
MODE = os.environ.get("BENCH_MODE", "B")
DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
CONFIGS = [
    ("8192x4096", 8192, 4096),
    ("4096x32000", 4096, 32000),
    ("16384x50304", 16384, 50304),
    ("2048x128256", 2048, 128256),
    ("4096x1024", 4096, 1024),
]
ITERS = int(os.environ.get("ITERS", "100"))


def bench(B, V, dt, backward):
    x = torch.randn(B, V, device=DEV, dtype=dt, requires_grad=backward)
    t = torch.randint(0, V, (B,), device=DEV)
    for _ in range(20):
        loss = F.cross_entropy(x, t)
        if backward:
            x.grad = None
            loss.backward()
    torch.mps.synchronize()
    s = time.perf_counter()
    for _ in range(ITERS):
        loss = F.cross_entropy(x, t)
        if backward:
            x.grad = None
            loss.backward()
    torch.mps.synchronize()
    return (time.perf_counter() - s) / ITERS * 1e6


forced = os.environ.get("PYTORCH_MPS_FORCE_DECOMPOSED_CROSS_ENTROPY")
sys.stderr.write(f"MODE={MODE} FORCE_DECOMP={forced!r} torch={torch.__file__}\n")
out = {}
for nm, B, V in CONFIGS:
    for dn, dt in DTYPES.items():
        for kind in ("fwd", "fwdbwd"):
            try:
                out[f"{nm}|{dn}|{kind}"] = round(bench(B, V, dt, kind == "fwdbwd"), 1)
            except Exception as e:
                out[f"{nm}|{dn}|{kind}"] = f"ERR:{str(e)[:50]}"
print(json.dumps(out))
