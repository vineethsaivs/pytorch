# Owner(s): ["module: inductor"]
import torch
import torch._functorch.aot_autograd as aot
import torch.utils._pytree as pytree
from torch._inductor import compile_to_python as inductor_compile_to_python
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import run_tests, TestCase


def _capture(m, x):
    """Capture m(x) as a flat-input ATen graph (params+buffers then x lifted to
    inputs), mirroring how torch.precompile feeds standalone graphs to a backend.

    NOTE: tracing runs m(x) once, which mutates m's buffers for stateful modules;
    callers should capture from a throwaway model and run on a fresh one.
    """
    pnames = [n for n, _ in m.named_parameters()]
    bnames = [n for n, _ in m.named_buffers()]
    pb = [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()]
    k = len(pnames)

    def flat_fn(flat):
        params = dict(zip(pnames, flat[:k]))
        buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
        with stateless._reparametrize_module(
            m, {**params, **buffers}, tie_weights=True
        ):
            out = m(flat[-1])
        leaves, _ = pytree.tree_flatten(out)
        return leaves

    with torch.enable_grad():
        gm = make_fx(flat_fn)(pb + [x])
    return gm


def _flat_inputs(m, x):
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


class TestInductorCompileToPython(TestCase):
    # torch._inductor.compile_to_python returns the INNER call only (no epilogue);
    # for a dense graph that is the whole computation, run under no_grad.
    def test_inner_call_dense_matches_eager(self):
        torch.manual_seed(0)
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)

        inner_src, cache = inductor_compile_to_python(gm, _flat_inputs(m, x))
        self.assertIsInstance(inner_src, str)
        self.assertIsNotNone(cache)  # non-mutating graph is serializable

        call = _exec(inner_src)
        with torch.no_grad():
            out = call(_flat_inputs(m, x))
        self.assertEqual(out[0], m(x))


class TestAotCompileToPython(TestCase):
    # torch._functorch.aot_autograd.compile_to_python returns the full self-contained
    # module (inner call + AOTAutograd's codegen'd epilogue).
    def test_dense_matches_eager(self):
        torch.manual_seed(0)
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)

        src, cache = aot.compile_to_python(gm, _flat_inputs(m, x))
        self.assertIsInstance(src, str)
        self.assertIsNotNone(cache)
        # The epilogue is AOTAutograd's own codegen, not a hand-rolled driver.
        self.assertIn("_runtime_wrapper", src)

        call = _exec(src)
        self.assertEqual(call(_flat_inputs(m, x))[0], m(x))

    def test_self_contained_runs_without_cache(self):
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        src, _cache = aot.compile_to_python(gm, _flat_inputs(m, x))
        call = _exec(src)
        self.assertEqual(call(_flat_inputs(m, x))[0], m(x))

    def test_buffer_mutation_is_reflected(self):
        # BatchNorm in training mutates running stats. The composed module reflects
        # that onto the passed-in buffers via AOTAutograd's captured orchestration,
        # and matches eager.
        def fresh():
            torch.manual_seed(0)
            mm = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            mm.train()
            return mm

        x = torch.randn(8, 4)
        gm = _capture(fresh(), x)  # throwaway model mutated during tracing
        src, _cache = aot.compile_to_python(gm, _flat_inputs(fresh(), x))

        ref = fresh()
        ref_out = ref(x)
        ref_rm = ref[1].running_mean.clone()
        ref_rv = ref[1].running_var.clone()
        ref_nbt = ref[1].num_batches_tracked.clone()

        run = fresh()
        call = _exec(src)
        out = call(_flat_inputs(run, x))

        self.assertEqual(out[0], ref_out)
        self.assertEqual(run[1].running_mean, ref_rm)
        self.assertEqual(run[1].running_var, ref_rv)
        self.assertEqual(run[1].num_batches_tracked, ref_nbt)
        self.assertNotEqual(float(run[1].running_mean.abs().sum()), 0.0)

    def test_duplicate_input(self):
        # The same tensor passed as two graph inputs goes through AOTAutograd's
        # dedup wrapper; the general composition handles it.
        t = torch.randn(4)

        def flat_fn(flat):
            return pytree.tree_flatten(flat[0] + flat[1])[0]

        with torch.enable_grad():
            gm = make_fx(flat_fn)([t, t])
        src, _cache = aot.compile_to_python(gm, [t, t])
        call = _exec(src)
        self.assertEqual(call([t, t])[0], t + t)

    def test_functionalized_rng(self):
        # Functionalized RNG (dropout) threads seed/offset through the calling
        # convention; the RNG wrapper composes in (the verification is structural,
        # since the two RNG paths draw different masks).
        import torch._functorch.config as functorch_config

        x = torch.randn(64)

        def flat_fn(flat):
            return pytree.tree_flatten(
                torch.nn.functional.dropout(flat[0], 0.5, training=True)
            )[0]

        with functorch_config.patch(functionalize_rng_ops=True):
            with torch.enable_grad():
                gm = make_fx(flat_fn)([x])
            src, _cache = aot.compile_to_python(gm, [x])
            self.assertIn("CUDARngStateHelper", src)
            call = _exec(src)
            out = call([x])
        self.assertEqual(out[0].shape, x.shape)
        # Dropout zeros some elements: a valid masked result, not the identity.
        self.assertTrue((out[0] == 0).any())

    def test_effectful_op_not_supported(self):
        # An effectful custom op makes the Inductor artifact non-saveable, so the
        # inner code cannot be extracted to standalone source -> clean error.
        from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
        from torch.library import _scoped_library

        with _scoped_library("mlctp", "FRAGMENT") as lib:
            lib.define("eff(Tensor x) -> Tensor")
            lib.impl("eff", lambda x: x + 1.0, "CompositeExplicitAutograd")
            lib.impl("eff", lambda x: torch.empty_like(x), "Meta")
            op = torch.ops.mlctp.eff.default
            _register_effectful_op(op, _EffectType.ORDERED)
            try:
                x = torch.randn(4)

                def flat_fn(flat):
                    return pytree.tree_flatten(torch.ops.mlctp.eff(flat[0]))[0]

                with torch.enable_grad():
                    gm = make_fx(flat_fn)([x])
                with self.assertRaises(NotImplementedError):
                    aot.compile_to_python(gm, [x])
            finally:
                _register_effectful_op(op, None)

    def test_output_alias(self):
        # An output that is a view of an input goes through AOTAutograd's output-
        # alias epilogue (gen_alias_from_base); the composed module reproduces it.
        x = torch.randn(3, 4)

        def fn(a):
            return a.t()

        def flat_fn(flat):
            return pytree.tree_flatten(fn(*flat))[0]

        with torch.enable_grad():
            gm = make_fx(flat_fn)([x])
        src, _cache = aot.compile_to_python(gm, [x])
        self.assertIn("gen_alias_from_base", src)
        # Runtime helpers are imported from the single stable runtime surface,
        # not scattered AOTAutograd internals.
        self.assertIn("standalone_runtime import gen_alias_from_base", src)

        call = _exec(src)
        out = call([x])
        self.assertEqual(out[0], x.t())

    def test_dtensor_subclass(self):
        # DTensor (tensor-subclass) graph inputs/outputs go through AOTAutograd's
        # subclass wrap/unwrap; the composed module reproduces it (the subclass
        # wrapper source is embedded and re-exec'd with pickle-embedded metadata).
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")
        import os

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        saved = {k: os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT")}
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))
            m = torch.nn.Linear(4, 3).eval()
            for name, p in list(m.named_parameters()):
                setattr(
                    m,
                    name,
                    torch.nn.Parameter(
                        distribute_tensor(p.detach(), mesh, [Replicate()])
                    ),
                )
            x = distribute_tensor(torch.randn(5, 4), mesh, [Replicate()])
            ref = m(x)

            gm = _capture(m, x)
            src, _cache = aot.compile_to_python(gm, _flat_inputs(m, x))
            self.assertIn("__tensor_unflatten__", src)

            call = _exec(src)
            out = call(_flat_inputs(m, x))
            self.assertEqual(out[0].to_local(), ref.to_local())
        finally:
            dist.destroy_process_group()
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    run_tests()
