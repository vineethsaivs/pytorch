# Owner(s): ["module: inductor"]

import torch
import torch._inductor.metrics as metrics
from torch._inductor import config
from torch._inductor.fx_passes.control_dependencies import (
    control_deps,
    FUSE_REGION,
    mark_fuse_region,
)
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.fx import Graph, GraphModule
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    requires_gpu,
)


class TestControlDeps(InductorTestCase):
    def test_fuse_region_subgraph_arg_fake_update(self):
        from torch._inductor.fx_utils import _extract_subgraphs_and_args

        def make_region() -> GraphModule:
            graph = Graph()
            x = graph.placeholder("x")
            nested = graph.placeholder("nested")
            add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
            mul = graph.call_function(torch.ops.aten.mul.Tensor, (add, 2))
            graph.output((mul, nested))
            return GraphModule({}, graph)

        def make_nested() -> GraphModule:
            graph = Graph()
            x = graph.placeholder("x")
            graph.output(graph.call_function(torch.ops.aten.sin.default, (x,)))
            return GraphModule({}, graph)

        region = make_region()
        nested = make_nested()
        outer_graph = Graph()
        node = outer_graph.call_function(control_deps, args=((), region), kwargs={})

        x = torch.empty(1)
        subgraphs = {region: None, nested: None}
        extracted = list(
            _extract_subgraphs_and_args(
                node,
                subgraphs,
                (),
                region,
                x,
                nested,
                **{FUSE_REGION: True},
            )
        )

        self.assertEqual(len(extracted), 1)
        self.assertIs(extracted[0][0], region)
        self.assertIs(extracted[0][1][0], x)
        self.assertIs(extracted[0][1][1], nested)

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_prevents_fusion(self):
        def fn(a, b):
            c = a + 1
            d = b @ b
            e = c * 2
            return d, e

        def add_control_deps(graph):
            nodes = [n for n in graph.nodes if n.op == "call_function"]
            if len(nodes) != 3:
                raise AssertionError(f"Expected 3 nodes, got {len(nodes)}")
            c_node = nodes[0]
            d_node = nodes[1]
            e_node = nodes[2]

            if d_node.target != torch.ops.aten.mm.default:
                raise AssertionError(f"Expected mm.default, got {d_node.target}")

            from torch.utils._ordered_set import OrderedSet

            deps_map = {d_node: OrderedSet([c_node]), e_node: OrderedSet([d_node])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            sub_g = graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.control_deps
            )
            if len(sub_g) != 2:
                raise AssertionError(f"Expected 2 control_deps nodes, got {len(sub_g)}")

            if list(sub_g[0].meta["val"].shape) != [256, 256]:
                raise AssertionError(
                    f"Expected shape [256, 256], got {list(sub_g[0].meta['val'].shape)}"
                )
            if list(sub_g[1].meta["val"].shape) != [256, 256]:
                raise AssertionError(
                    f"Expected shape [256, 256], got {list(sub_g[1].meta['val'].shape)}"
                )

            for attr in graph.find_nodes(op="get_attr"):
                for n in getattr(graph.owning_module, attr.target).graph.nodes:
                    if list(n.meta["val"].shape) != [256, 256]:
                        raise AssertionError(
                            f"Expected shape [256, 256], got {list(n.meta['val'].shape)}"
                        )

            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            compiled_fn = torch.compile(fn)
            a = torch.rand([256, 256], device=GPU_TYPE)
            b = torch.rand([256, 256], device=GPU_TYPE)

            _, code = run_and_get_code(torch.compile(fn), a, b)
            result = compiled_fn(a, b)

            FileCheck().check(".run(").check("extern_kernels.mm(").check(".run(").run(
                code[0]
            )

            expected = fn(a, b)
            torch.testing.assert_close(result, expected)

    @config.patch(allow_buffer_reuse=False)
    @requires_gpu()
    def test_control_deps_do_not_extend_buffer_lifetime(self):
        """
        Control deps should not extend buffer lifetimes - buf0/buf1 should be
        deleted before the 4th matmul, not kept alive by the control dependency.
        """

        def fn(a, b):
            # Chain of 4 matmuls: mm0 -> mm1 -> mm2 -> mm3
            mm0 = a @ b
            mm1 = mm0 @ b
            mm2 = mm1 @ b
            mm3 = mm2 @ b
            return mm3

        def add_control_deps(graph):
            from torch.utils._ordered_set import OrderedSet

            mm_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mm.default
            )
            if len(mm_nodes) != 4:
                raise AssertionError(f"Expected 4 mm nodes, got {len(mm_nodes)}")

            # Add control dep: mm3 depends on mm0's output
            # This should NOT extend mm0's buffer lifetime
            deps_map = {mm_nodes[3]: OrderedSet([mm_nodes[0]])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            a = torch.rand([256, 256], device=GPU_TYPE)
            b = torch.rand([256, 256], device=GPU_TYPE)

            result, code = run_and_get_code(torch.compile(fn), a, b)
            torch.testing.assert_close(result, fn(a, b))

            # buf0 should be allocated, passed in out=, used once, then del
            FileCheck().check("buf0 = ").check_count(
                "extern_kernels.mm", 2, exactly=True
            ).check("del buf0").run(code[0])

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_with_nested_args(self):
        """Test control_deps with operations that have nested args (e.g., torch.cat)."""

        def fn(a, b, c):
            x = a + 1
            y = b * 2
            # torch.cat has nested args: (List[Tensor], dim)
            cat_result = torch.cat([x, y], dim=0)
            z = cat_result + c
            return z

        def add_control_deps(graph):
            from torch.utils._ordered_set import OrderedSet

            # Find the cat node which has nested args
            cat_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.cat.default
            )
            if len(cat_nodes) != 1:
                raise AssertionError(f"Expected 1 cat node, got {len(cat_nodes)}")
            cat_node = cat_nodes[0]

            # Verify it has nested args (list of tensors)
            if not isinstance(cat_node.args[0], (list, tuple)):
                raise AssertionError(
                    f"Expected nested args, got {type(cat_node.args[0])}"
                )

            # Find a node that comes before cat to use as dependency
            add_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.add.Tensor
            )
            # Use the first add node (x = a + 1)
            dep_node = add_nodes[0]

            deps_map = {cat_node: OrderedSet([dep_node])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )

            # Verify control_deps was created
            control_deps_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.control_deps
            )
            if len(control_deps_nodes) != 1:
                raise AssertionError(
                    f"Expected 1 control_deps node, got {len(control_deps_nodes)}"
                )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            a = torch.rand([128, 64], device=GPU_TYPE)
            b = torch.rand([128, 64], device=GPU_TYPE)
            c = torch.rand([256, 64], device=GPU_TYPE)

            compiled_fn = torch.compile(fn)
            result = compiled_fn(a, b, c)

            expected = fn(a, b, c)
            torch.testing.assert_close(result, expected)

    @config.patch(enable_auto_functionalized_v2=True)
    def test_control_deps_with_auto_functionalized_v2(self):
        with torch.library._scoped_library("control_deps_auto_func", "FRAGMENT") as lib:
            lib.define(
                "add_one_(Tensor(a!) x) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
            )

            def add_one_impl(x):
                x.add_(1.0)
                return x.clone()

            lib.impl("add_one_", add_one_impl, "CompositeExplicitAutograd")

            @torch.library.register_fake("control_deps_auto_func::add_one_", lib=lib)
            def add_one_fake(x):
                return x.clone()

            def fn(x):
                y = x * 2
                z = torch.ops.control_deps_auto_func.add_one_(x)
                return z + y

            def add_control_deps(graph):
                from torch.utils._ordered_set import OrderedSet

                auto_func_nodes = graph.find_nodes(
                    op="call_function",
                    target=torch.ops.higher_order.auto_functionalized_v2,
                )
                if len(auto_func_nodes) != 1:
                    raise AssertionError(
                        f"Expected 1 auto_functionalized_v2 node, got {len(auto_func_nodes)}"
                    )

                mul_nodes = graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mul.Tensor
                )
                if len(mul_nodes) != 1:
                    raise AssertionError(f"Expected 1 mul node, got {len(mul_nodes)}")

                torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                    graph, {auto_func_nodes[0]: OrderedSet([mul_nodes[0]])}
                )
                control_deps_nodes = graph.find_nodes(
                    op="call_function", target=torch.ops.higher_order.control_deps
                )
                if len(control_deps_nodes) != 1:
                    raise AssertionError(
                        f"Expected 1 control_deps node, got {len(control_deps_nodes)}"
                    )

                subgraph_attr = control_deps_nodes[0].args[1]
                if not isinstance(subgraph_attr, torch.fx.Node):
                    raise AssertionError(
                        f"Expected get_attr node for subgraph, got {type(subgraph_attr)}"
                    )
                subgraph = getattr(graph.owning_module, subgraph_attr.target)
                nested_auto_func_nodes = subgraph.graph.find_nodes(
                    op="call_function",
                    target=torch.ops.higher_order.auto_functionalized_v2,
                )
                if len(nested_auto_func_nodes) != 1:
                    raise AssertionError(
                        "Expected control_deps subgraph to contain "
                        f"1 auto_functionalized_v2 node, got {len(nested_auto_func_nodes)}"
                    )
                return graph

            x = torch.zeros(8)
            compiled_x = x.clone()
            eager_x = x.clone()

            with torch._inductor.config.patch(
                post_grad_custom_post_pass=add_control_deps,
            ):
                result = torch.compile(fn, backend="inductor", fullgraph=True)(
                    compiled_x
                )

            expected = fn(eager_x)
            torch.testing.assert_close(result, expected)
            torch.testing.assert_close(compiled_x, eager_x)

    @requires_gpu()
    def test_control_deps_with_triton_kernel(self):
        """Test control_deps with triton_kernel_wrapper_mutation."""
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        def fn(x, y):
            z = x * 2
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output + z

        def add_control_deps(graph):
            from torch.utils._ordered_set import OrderedSet

            # Find triton_kernel_wrapper_mutation nodes
            triton_nodes = graph.find_nodes(
                op="call_function",
                target=torch.ops.higher_order.triton_kernel_wrapper_functional,
            )
            if not triton_nodes:
                raise AssertionError("Expected triton_kernel_wrapper_functional nodes")
            # Find mul node (z = x * 2) to use as dependency
            mul_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )
            if not mul_nodes:
                raise AssertionError("Expected mul.Tensor nodes")
            deps_map = {triton_nodes[0]: OrderedSet([mul_nodes[0]])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            x = torch.rand([256], device=GPU_TYPE)
            y = torch.rand([256], device=GPU_TYPE)

            compiled_fn = torch.compile(fn)
            result = compiled_fn(x, y)

            expected = fn(x, y)
            torch.testing.assert_close(result, expected)

    @requires_gpu()
    def test_control_deps_orders_void_op_across_nested_calls(self):
        """record_event's void op must be named as an additional_buffer_dep
        of the subsequent wait_event's operations after Inductor lowering.

        record_event lowers to a NoneLayout (void) op and the overall
        control_deps call around it returns a tuple/None.  When a later
        control_deps call (around wait_event) lists the record's control_deps
        node as an additional dep, the lowered value fails the
        ``isinstance(dep, IRNode)`` check.  Before the fix, the void op was
        silently dropped and never referenced in the wait's
        additional_buffer_deps, so Inductor's cudagraph partitioning and
        other consumers of additional_buffer_deps could reorder the wait
        before the record.
        """
        from torch._inductor import ir
        from torch._inductor.virtualized import V

        def fn(x):
            s1 = torch.Stream(device=GPU_TYPE)
            s2 = torch.Stream(device=GPU_TYPE)
            e = torch.Event()
            with s1:
                y = x + 1
                e.record()
            with s2:
                a = x * 3
                e.wait()
                z = y * a
            return z

        captured: list[dict] = []

        def capture(nodes):
            void_names = {
                op.get_name()
                for op in V.graph.operations
                if isinstance(op, ir.Buffer) and isinstance(op.layout, ir.NoneLayout)
            }
            referenced: set[str] = set()
            for deps in V.graph.additional_buffer_deps.values():
                referenced.update(deps)
            captured.append({"void_names": void_names, "referenced": referenced})
            return nodes

        torch._dynamo.reset()
        with config.patch(_pre_fusion_custom_pass=capture):
            x = torch.ones(2, 2, device=GPU_TYPE)
            torch.compile(fn)(x)

        self.assertTrue(captured, "expected at least one Inductor compile")

        void_names: set[str] = set()
        referenced: set[str] = set()
        for state in captured:
            void_names |= state["void_names"]
            referenced |= state["referenced"]

        self.assertGreater(
            len(void_names),
            0,
            "expected record_event/wait_event to lower to NoneLayout ops",
        )
        self.assertTrue(
            void_names & referenced,
            "no record_event void op appears as an additional_buffer_dep; "
            f"void_names={void_names}, referenced={referenced}",
        )

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_fuse_region_single_region(self):
        def fn(x):
            before = x + 1
            region0 = x * 2
            region1 = torch.relu(region0)
            after = x + 3
            return before, region1, after

        def add_fuse_region(graph):
            mul_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )[0]
            relu_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.relu.default
            )[0]
            mark_fuse_region(graph, [mul_node, relu_node])
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_fuse_region,
        ):
            x = torch.rand([256, 256], device=GPU_TYPE)
            result, code = run_and_get_code(torch.compile(fn), x)

        torch.testing.assert_close(result, fn(x))
        FileCheck().check("fused_add").check("fused_mul_relu").run(code[0])

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_fuse_region_keeps_outside_fusable(self):
        def fn(x):
            before = x + 1
            region0 = x * 2
            region1 = torch.relu(region0)
            after = x + 3
            return before + after, region1

        def add_fuse_region(graph):
            mul_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )[0]
            relu_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.relu.default
            )[0]
            mark_fuse_region(graph, [mul_node, relu_node])
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_fuse_region,
        ):
            x = torch.rand([256, 256], device=GPU_TYPE)
            result, code = run_and_get_code(torch.compile(fn), x)

        torch.testing.assert_close(result, fn(x))
        FileCheck().check(
            "Topologically Sorted Source Nodes: [before, after, add_2]"
        ).check("fused_mul_relu").run(code[0])
        FileCheck().check_not("fused_add_mul_relu").run(code[0])

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_fuse_region_multiple_regions(self):
        def fn(x):
            region0 = torch.relu(x * 2)
            region1 = torch.sigmoid(x + 3)
            return region0, region1

        def add_fuse_regions(graph):
            mul_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )[0]
            relu_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.relu.default
            )[0]
            add_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.add.Tensor
            )[0]
            sigmoid_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.sigmoid.default
            )[0]
            mark_fuse_region(graph, [mul_node, relu_node])
            mark_fuse_region(graph, [add_node, sigmoid_node])
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_fuse_regions,
        ):
            x = torch.rand([256, 256], device=GPU_TYPE)
            result, code = run_and_get_code(torch.compile(fn), x)

        torch.testing.assert_close(result, fn(x))
        FileCheck().check("fused_mul_relu").check("fused_add_sigmoid").run(code[0])
        FileCheck().check_not("fused_add_mul_relu_sigmoid").run(code[0])

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_fuse_region_multiple_outputs(self):
        def fn(x):
            region0 = x * 2
            region1 = torch.relu(region0)
            region2 = x + 3
            region3 = torch.sigmoid(region2)
            outside = x - 4
            return region1 + outside, region3 * outside

        def add_fuse_region(graph):
            mul_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )[0]
            relu_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.relu.default
            )[0]
            add_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.add.Tensor
            )[0]
            sigmoid_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.sigmoid.default
            )[0]
            mark_fuse_region(graph, [mul_node, relu_node, add_node, sigmoid_node])
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_fuse_region,
        ):
            x = torch.rand([256, 256], device=GPU_TYPE)
            result, code = run_and_get_code(torch.compile(fn), x)

        torch.testing.assert_close(result, fn(x))
        FileCheck().check("fused_add_mul_relu_sigmoid").run(code[0])

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_fuse_region_interleaved_input(self):
        def fn(x):
            region0 = x * 2
            outside = x + 3
            region1 = region0 + outside
            region2 = torch.relu(region1)
            after = x - 4
            return region2, after

        def add_fuse_region(graph):
            mul_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )[0]
            add_node = next(
                node
                for node in graph.find_nodes(
                    op="call_function", target=torch.ops.aten.add.Tensor
                )
                if mul_node in node.all_input_nodes
            )
            relu_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.relu.default
            )[0]
            mark_fuse_region(graph, [mul_node, add_node, relu_node])
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_fuse_region,
        ):
            x = torch.rand([256, 256], device=GPU_TYPE)
            result, code = run_and_get_code(torch.compile(fn), x)

        torch.testing.assert_close(result, fn(x))
        FileCheck().check("fused_add_sub").check("fused_add_mul_relu").run(code[0])

    def test_mark_fuse_region_rejects_boundary_cycle(self):
        def fn(x):
            region0 = x * 2
            outside = region0 + 3
            region1 = torch.relu(outside)
            return region1

        gm = torch.fx.symbolic_trace(fn)
        mul_node = next(node for node in gm.graph.nodes if node.name == "mul")
        relu_node = next(node for node in gm.graph.nodes if node.name == "relu")

        with self.assertRaisesRegex(AssertionError, "acyclic"):
            mark_fuse_region(gm.graph, [mul_node, relu_node])

    def test_mark_fuse_region_allows_no_external_outputs(self):
        def fn(x):
            _ = x + 1
            return x

        gm = torch.fx.symbolic_trace(fn)
        add_node = next(node for node in gm.graph.nodes if node.name == "add")

        region_node = mark_fuse_region(gm.graph, [add_node])
        gm.recompile()

        self.assertEqual(region_node.target, control_deps)
        self.assertIs(region_node.kwargs.get(FUSE_REGION), True)
        self.assertEqual(region_node.meta["val"], ())

        get_subgraph = region_node.args[1]
        subgraph = getattr(gm, get_subgraph.target)
        output_node = next(node for node in subgraph.graph.nodes if node.op == "output")
        self.assertEqual(output_node.args[0], ())

        x = torch.ones(2, 2)
        self.assertEqual(gm(x), fn(x))

    @config.patch(
        {
            "reorder_for_locality": False,
            "combo_kernels": True,
            "benchmark_combo_kernel": False,
            "combo_kernel_max_distance": -1,
            "combo_kernel_peak_memory_increase_gb": None,
            "combo_kernel_peak_memory_pct_threshold": None,
        }
    )
    @requires_gpu()
    def test_control_deps_fuse_region_blocks_combo_kernel(self):
        def fn(x):
            region = torch.relu(x * 2)
            outside = torch.sigmoid(x + 3)
            return region, outside

        def add_fuse_region(graph):
            mul_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )[0]
            relu_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.relu.default
            )[0]
            mark_fuse_region(graph, [mul_node, relu_node])
            return graph

        metrics.reset()
        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_fuse_region,
        ):
            x = torch.rand([256, 256], device=GPU_TYPE)
            result, code = run_and_get_code(torch.compile(fn), x)

        torch.testing.assert_close(result, fn(x))
        self.assertEqual(metrics.generated_kernel_count, 2)
        FileCheck().check("fused_mul_relu").check("fused_add_sigmoid").run(code[0])


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
