# Capture recipe

Author a throwaway capture script in `agent_space/` that wraps one step in
`DebugMode` and writes a per-op fingerprint log. Adapt it to the model /
training loop. Nothing here is committed.

Everything below was verified end-to-end on torch 2.10 (CUDA) and the API
points re-checked on torch 2.13. `torch.utils._debug_mode` is private and
version-dependent, so **start with the probe**.

## Step 1: probe the operator stream (do this first, every time)

The capture hinges on how `dm.operators` represents module boundaries, which
changes across torch versions. Confirm it before writing the capture:

```python
import torch, torch.nn as nn
from torch.utils._debug_mode import DebugMode

class Blk(nn.Module):
    def __init__(s): super().__init__(); s.fc = nn.Linear(64, 64)
    def forward(s, x): return torch.relu(s.fc(x))

m = nn.Sequential(Blk(), Blk())
dm = DebugMode(record_realtensor=True, record_nn_module=True, record_ids=True)
with dm, DebugMode.log_tensor_hashes(hash_fn="norm", hash_inputs=True):
    m(torch.randn(8, 64)).sum().backward()

for o in dm.operators:
    t = type(o).__name__
    if t == "_OpCall":
        print("op  d=%s %s | log=%s" % (o.call_depth, o.op, o.log))
    else:
        attrs = {a: getattr(o, a, None) for a in dir(o)
                 if not a.startswith("__") and not callable(getattr(o, a, None))}
        print(t, attrs)
```

What this tells you:

- The op-call class is `_OpCall` (import: `from torch.utils._debug_mode import
  _OpCall`). Each has `.op` (the `OpOverload`), `.call_depth`, `.record` (your
  dispatch-hook dict), and `.log` (the hash dict, e.g.
  `{'hash': 446.47, 'input_hash': ((7.69, 377.1, 508.4), {})}`).
- The **module-marker class and its FQN attribute vary by version**. Observed:
  - torch ~2.10: class `_NNModuleCall`, FQN in `.module_name`
    (e.g. `Net.blocks.0.fc1`), with `.call_depth`.
  - torch ~2.13: class `_AnnotateCall` with `header == "nn.Mod"`, FQN in
    `.tag` (e.g. `Sequential.0.fc`), with `.call_depth`.
  Set `MODULE_MARKER` / the FQN getter in the capture below to whatever the
  probe shows.
- Module ops appear at `call_depth = marker_depth + 1`, right after their
  marker. Backward ops all sit at `call_depth == 1` with **no** module markers
  -> they need the grad_fn map (below).

## Step 2: the capture function (verified working)

```python
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.utils._debug_mode import DebugMode, _OpCall

# --- set these from the probe output ---
def is_module_marker(o):
    t = type(o).__name__
    return t == "_NNModuleCall" or (t == "_AnnotateCall" and getattr(o, "header", None) == "nn.Mod")

def marker_fqn(o):
    return getattr(o, "module_name", None) or getattr(o, "tag", None)
# ----------------------------------------

EXCLUDED = frozenset({
    # no real compute / just rewrap storage -> drop so the diff isn't polluted
    "t", "transpose", "view", "_unsafe_view", "reshape", "permute", "expand",
    "squeeze", "unsqueeze", "detach", "clone", "_to_copy", "to", "contiguous",
    "ones_like", "zeros_like", "empty_like", "slice", "select", "as_strided",
})
FLOAT = {torch.float32, torch.float16, torch.bfloat16}

def _op_name(func):
    return func._schema.name.split("::")[-1] if hasattr(func, "_schema") else str(func).split(".")[-1]

def _float_leaves(result, min_numel):
    out = []
    for leaf in pytree.tree_leaves(result):
        if isinstance(leaf, torch.Tensor) and not isinstance(leaf, torch._subclasses.FakeTensor):
            if leaf.numel() >= min_numel and leaf.dtype in FLOAT:
                out.append(leaf)
    return out

def _stats(t):
    t = t.detach(); t64 = t.to(torch.float64)
    return {
        "Shape": f"{tuple(t.shape)} {t.dtype}",
        "L2": f"{t64.norm(2).item():.6e}",
        "Min": f"{t.float().min().item():.6e}",
        "Max": f"{t.float().max().item():.6e}",
        "Mean": f"{t64.mean().item():.6e}",
    }

def _hashstr(h):
    vals = [v for v in pytree.tree_leaves(h) if v is not None]
    return ", ".join(f"{v:.6e}" if isinstance(v, float) else str(v) for v in vals)

def capture(model, run_fn, *, min_numel=0):
    """run_fn(model) executes one forward (+backward). Returns ordered list of
    {key, phase, stats, out_hash, in_hash}."""
    # backward FQN map: walk the autograd graph from each module's outputs
    grad_fn_to_fqn = {}
    id_to_fqn = {id(m): n for n, m in model.named_modules()}
    pending = {}
    def _gfns(obj):
        return {a.grad_fn for a in pytree.tree_leaves(obj)
                if isinstance(a, torch.Tensor) and a.grad_fn is not None}
    def pre(mod, args):
        if id(mod) in id_to_fqn: pending[id(mod)] = _gfns(args)
    def post(mod, args, output):
        fqn = id_to_fqn.get(id(mod))
        if fqn is None: return
        inputs = pending.pop(id(mod), set())
        for t in pytree.tree_leaves(output):
            if not isinstance(t, torch.Tensor) or t.grad_fn is None: continue
            stack, seen = [t.grad_fn], set()
            while stack:
                fn = stack.pop()
                if fn in seen or fn in inputs: continue
                seen.add(fn)
                grad_fn_to_fqn.setdefault(fn, fqn)   # don't overwrite child claims
                for parent, _ in fn.next_functions:
                    if parent is not None and parent not in seen: stack.append(parent)
    h1 = nn.modules.module.register_module_forward_pre_hook(pre)
    h2 = nn.modules.module.register_module_forward_hook(post)

    # phase + backward-fqn + inline stats must be captured at dispatch time
    def record_hook(func, types, args, kwargs, result):
        node = torch._C._current_autograd_node()
        rec = {"phase": "backward" if node is not None else "forward",
               "bwd_fqn": grad_fn_to_fqn.get(node) if node is not None else None}
        leaves = _float_leaves(result, min_numel)
        if leaves: rec["stats"] = _stats(leaves[0])
        return rec

    dm = DebugMode(record_realtensor=True, record_nn_module=True, record_ids=True)
    try:
        with dm, DebugMode.log_tensor_hashes(hash_fn="norm", hash_inputs=True), \
                DebugMode.dispatch_hooks(record_hook=record_hook):
            run_fn(model)
    finally:
        h1.remove(); h2.remove()

    # forward FQN: reconstruct from the marker/call_depth stream with a
    # depth-truncated stack (pop markers at depth >= current before handling)
    out, mod_stack, counters, root = [], [], {}, None
    for o in dm.operators:
        depth = o.call_depth
        while mod_stack and mod_stack[-1][0] >= depth:
            mod_stack.pop()
        if is_module_marker(o):
            fqn = marker_fqn(o)
            if root is None: root = fqn
            mod_stack.append((depth, fqn))
            continue
        if not isinstance(o, _OpCall): continue
        name = _op_name(o.op)
        if name in EXCLUDED: continue
        rec, log = o.record or {}, o.log or {}
        phase = rec.get("phase", "forward")
        fqn = (rec.get("bwd_fqn") or "<backward>") if phase == "backward" else \
              (mod_stack[-1][1] if mod_stack else "<none>")
        if root and fqn.startswith(root + "."): fqn = fqn[len(root) + 1:]
        elif fqn == root: fqn = "<root>"
        out_hash, in_hash = _hashstr(log.get("hash")), _hashstr(log.get("input_hash"))
        if "stats" not in rec and not out_hash and not in_hash: continue
        seq = counters.get(fqn, 0); counters[fqn] = seq + 1
        out.append({"key": f"{fqn}/op_{seq}_{name}", "phase": phase,
                    "stats": rec.get("stats", {}), "out_hash": out_hash, "in_hash": in_hash})
    return out

def write_log(captures, path):
    with open(path, "w") as f:
        f.write(f"Total: {len(captures)}\n" + "=" * 60 + "\n\n")
        for c in captures:
            f.write(f"[{c['key']}]\n")
            for k, v in c["stats"].items(): f.write(f"  {k}: {v}\n")
            if c["out_hash"]: f.write(f"  Output hash: {c['out_hash']}\n")
            if c["in_hash"]:  f.write(f"  Input hashes: {c['in_hash']}\n")
            if c["phase"] != "forward": f.write(f"  Phase: {c['phase']}\n")
            f.write("\n")
```

Drive it once per run, with identical setup on both sides:

```python
def build():
    torch.manual_seed(0)
    return Net().cuda(), torch.randn(16, 256, device="cuda")

def step(m, x):
    m(x).sum().backward()

m, x = build()
write_log(capture(m, lambda mm: step(mm, x)), "run_A.log")
# ... change the one thing under test (compile, FSDP, a flag) ...
m, x = build()
write_log(capture(m, lambda mm: step(mm, x)), "run_B.log")
```

## Backward FQN attribution

During backward the C++ autograd engine drives execution, so no
`nn.Module.forward` runs and the stream has no module markers for backward ops
(they all sit at `call_depth == 1`). The `grad_fn_to_fqn` map above fixes this:
global forward hooks walk the autograd graph from each module's outputs and
claim unclaimed nodes for that module (`setdefault`, so a parent picks up glue
ops between children while children keep their own). This was verified to
recover e.g. `blocks.1.fc2` for a backward `mm`. Ops it can't claim fall back
to `<backward>`.

## Compiled / traced runs

Under `torch.compile`/`aot_fx_trace` the graph runs as `gm(*flat_inputs)`,
bypassing `nn.Module.forward`, so the marker stream is empty and ops land under
`<none>`/`<backward>`. Two options:

- If the traced `GraphModule` carries node metadata
  (`node.meta["custom"]["module_fqn"]`, `node.meta["stack_trace"]`,
  `node.meta.get("autograd_backward")`), replay via a `torch.fx.Interpreter`
  subclass that sets that context per node so the capture sees the same FQN /
  phase eager would. (Newer torch exposes `run_compile_with_interpreter` on
  `DebugMode` for related plumbing -- check the probe's init signature.)
- Otherwise, compare **by op sequence**: both logs in dispatch order, paired
  positionally, relying on op-name + shape + hash. FQN columns will be
  `<none>` but the divergence point is still findable.

## Customizing

- **`EXCLUDED`** — add an op that pollutes the diff with structurally
  identical rows; remove one to gain visibility. Collective comms
  (`all_gather_into_tensor`, `reduce_scatter_tensor`, `wait_tensor`) are
  deliberately *not* excluded — they often diverge between eager and traced.
- **`min_numel`** — raise to drop small tensors (set 0 to capture everything,
  as in the verified demo). Restrict to specific ops by matching `name`
  against an allowlist before the `EXCLUDED` check.
- **`hash_fn`** — `"norm"` (L1, float64; tolerance-friendly), `"hash_tensor"`
  (`torch.hash_tensor`, XOR-reduce of raw bytes, strict bit-exactness), a
  callable, or a list of the above for a tuple of hashes.
- **Scalars** (e.g. the loss `sum`) get an empty output hash — expected; they
  carry stats but nothing to hash meaningfully.
