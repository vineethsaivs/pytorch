---
name: numerics-debugging
description: Debug numeric / bitwise divergence between two PyTorch runs that should agree (eager vs torch.compile, eager vs aot_fx_trace, FSDP vs single-GPU, TF32 vs fp32, before vs after a refactor) by capturing per-op activations with torch.utils._debug_mode.DebugMode and diffing them to localize the first diverging op. Use when the user reports loss drift, mismatched outputs, "numerics don't match", "bitwise divergence", "find which op diverges", or invokes /numerics-debugging.
---

# Numerics Debugging (DebugMode-based)

When two runs that *should* produce the same numbers don't, the goal is to
find the **first op where they diverge**, then decide whether that op is the
cause (its inputs matched but its output didn't) or just a carrier (its inputs
already differed). This skill captures a per-op fingerprint of one step from
each run with `torch.utils._debug_mode.DebugMode`, then diffs the two.

There is no bundled tooling. You author a small throwaway capture script in
`agent_space/` from the verified recipe in
[references/capture-recipe.md](references/capture-recipe.md), run it once per
side, and compare the two logs.

## IMPORTANT: probe before you trust the recipe

`torch.utils._debug_mode` is **private and version-dependent**. The recipe was
verified on torch 2.10 and 2.13, but the operator-stream structure (in
particular how module boundaries are represented) changes across versions:

- torch ~2.10: module markers are `_NNModuleCall` with a `.module_name` attr.
- torch ~2.13: module markers are `_AnnotateCall` with `header == "nn.Mod"`
  and the FQN in `.tag`.
- `DebugMode.current_nn_module_stack` is **empty at dispatch time** in these
  versions, so you cannot read the FQN from inside the dispatch hook.

So the first step is always to run the ~15-line probe in the recipe against
the *actual* environment, confirm the marker class / attribute names, and
adapt the capture to match. Do not assume.

## Workflow

1. **Make the two runs comparable.** They MUST share dtype, seed, and inputs,
   and run deterministically. A precision change (bf16 vs fp32) makes every op
   diverge and the diff is useless. Set the same `torch.manual_seed(...)`
   before model init and before the step, use the same batch, and enable
   `torch.use_deterministic_algorithms(True)` where possible.
2. **Probe** the operator stream (recipe, step 1) to confirm the module-marker
   class for this torch version.
3. **Capture one step per run** with the capture function (recipe, step 2),
   writing each run's fingerprints to its own log (`run_A.log`, `run_B.log`).
   Capture on a warmed-up step if step 0 has legitimate one-time init/compile
   differences.
4. **Diff the two logs** (recipe, step 3 + [references/comparing-runs.md](references/comparing-runs.md)):
   pair ops by key, parse the hashes as floats, and report the first op whose
   output hash differs beyond a relative tolerance.

## What gets captured per op

For each non-excluded op producing a float tensor:

- **Key** `module_fqn/op_N_opname` (e.g. `blocks.0.fc1/op_0_addmm`) — FQN +
  per-module op counter + ATen op name.
- **Phase** — forward or backward (from `torch._C._current_autograd_node()`).
- **Output hash** — DebugMode's `norm` hash (~L1, float64) of the result.
- **Input hashes** — `norm` hash of each input *before* the op ran. This is
  what catches in-place mutations (e.g. `_fused_adam_` returns `None`, so it
  has no output hash, but its input hashes still move).
- **Stats** — shape/dtype + L2/min/max/mean, with L2 and mean in **float64**
  so they don't wobble with reduction order.

## Interpreting the diff

- **All hashes identical** -> the runs are bitwise equal; the bug is elsewhere
  (data loading, loss, optimizer state, RNG consumption order).
- **First divergence at op K, and K's input hashes already differ** -> the
  real divergence is upstream; follow the inputs back to their producing op
  and keep walking until you find an op whose **inputs match but output
  differs**. That op is the root cause.
- **Everything diverges from the very first op** -> almost always a dtype or
  seed/determinism mismatch, not a real bug. Fix setup (step 1) and recapture.
- **A tiny last-digit difference that doesn't propagate** -> reduction-order
  noise, not a bug. Compare the float64 L2/mean and the `norm` hash with a
  relative tolerance rather than requiring exact string equality.

This was validated on a real divergence: fp32 vs TF32 matmul on the same model
matched on every forward op but first diverged on a **backward** `mm`, whose
inputs matched -> correctly fingering the TF32 matmul kernel itself as the
source. Divergence often first crosses the hash's precision floor in backward
(larger reductions) even when forward already differs in lower digits.

## Cost

Capture adds roughly 10-40% time/memory on the single captured step only
(stats reduced inline in float64; no tensor clones held). Because the context
wraps just one step, steady-state training is untouched.

## References

- [references/capture-recipe.md](references/capture-recipe.md) — the probe,
  the verified `DebugMode` capture function (forward FQN via the marker/depth
  stream, backward FQN via a grad_fn map), op filtering, and the
  compiled/traced-graph caveat.
- [references/comparing-runs.md](references/comparing-runs.md) — the real log
  format, tolerance-based pairing/diffing, and the common eager-vs-traced
  key-drift patterns.
