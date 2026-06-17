# mypy: allow-untyped-defs
"""
Effect-ordering and fusion-region helpers for Inductor.

control_deps has two compiler-facing uses:

1. Ordering.  ``additional_deps`` contains tensors or FX nodes whose producers
   must run before the wrapped subgraph.  The wrapped subgraph computes the
   value normally, while the extra operands make an otherwise implicit ordering
   edge visible to Inductor.  This is used for stream/event synchronization and
   other side-effect-like operations that must not be reordered past producers.

2. Fusion regions.  ``fuse_region=True`` marks the operations created by the
   subgraph as one fusion region.  The region can fuse internally, but it cannot
   fuse with operations outside the region or with another region.  This is an
   Inductor scheduling constraint only; eager, fake, and autograd execution all
   behave as if the HOP directly called ``subgraph(*args, **kwargs)``.

These two uses can be combined, but they are independent.  Empty
``additional_deps`` is only meaningful for the fusion-region use case.
"""

from operator import attrgetter, getitem
from typing import Any

import torch.fx as fx
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.utils._ordered_set import OrderedSet


FUSE_REGION = "fuse_region"


class ControlDeps(HigherOrderOperator):
    """
    Higher-order operator for Inductor-only ordering and fusion constraints.

    Schema: control_deps(additional_deps, subgraph, *args, **kwargs) -> result
    where:
    - additional_deps: tuple/list of values whose producers must run first
    - subgraph: GraphModule or callable containing the operation(s) to execute
    - args/kwargs: arguments for the subgraph
    - fuse_region: if true, isolate subgraph operations from outside fusion

    The runtime value is exactly the subgraph result.  ``additional_deps`` and
    ``fuse_region`` exist to communicate constraints to Inductor lowering
    and scheduling, not to change eager semantics.
    """

    def __init__(self) -> None:
        super().__init__("control_deps")

    def __call__(
        self,
        additional_deps,
        subgraph,
        *args,
        fuse_region: bool = False,
        **kwargs,
    ):
        """Call the operator with dependencies and subgraph.

        Args:
            additional_deps: Tuple of tensors that must be computed first
            subgraph: GraphModule containing the exact operation to execute
            *args: Arguments to pass to the subgraph
        """
        if not isinstance(fuse_region, bool):
            raise TypeError(
                f"fuse_region must be bool, got {type(fuse_region).__name__}"
            )
        if not isinstance(additional_deps, (tuple, list)):
            raise TypeError(
                f"additional_deps must be tuple/list, got {type(additional_deps).__name__}"
            )
        if not (isinstance(subgraph, fx.GraphModule) or callable(subgraph)):
            raise TypeError(
                f"subgraph must be GraphModule or callable, got {type(subgraph).__name__}"
            )
        # pyrefly: ignore [missing-attribute]
        if fuse_region:
            return super().__call__(
                additional_deps,
                subgraph,
                *args,
                fuse_region=True,
                **kwargs,
            )
        return super().__call__(additional_deps, subgraph, *args, **kwargs)


control_deps = ControlDeps()

# control_deps wraps side-effecting ops (e.g. record_event, wait_event)
# and must not be eliminated by DCE even when its outputs are unused.
from torch.fx.node import has_side_effect


has_side_effect(control_deps)


# Register fake implementation for tracing
@register_fake(control_deps)
def _(additional_deps, subgraph, *args, fuse_region=False, **kwargs):
    """Fake tensor implementation - execute the subgraph."""
    return subgraph(*args, **kwargs)


# Register eager execution implementation
@control_deps.py_impl(DispatchKey.CompositeExplicitAutograd)
def control_deps_eager(additional_deps, subgraph, *args, fuse_region=False, **kwargs):
    """Eager implementation - just execute the subgraph."""
    return subgraph(*args, **kwargs)


# Autograd impl needed because additional_deps tensors may have autograd state,
# causing dispatch through AutogradCUDA even in post-autograd graphs.
@control_deps.py_impl(DispatchKey.Autograd)
def control_deps_autograd(
    additional_deps, subgraph, *args, fuse_region=False, **kwargs
):
    return subgraph(*args, **kwargs)


def get_subgraph_name(gm: fx.GraphModule, name):
    name = f"subgraph_{name}"

    if not hasattr(gm, name):
        return name

    i = 0
    while hasattr(gm, f"{name}_{i}"):
        i += 1

    return f"{name}_{i}"


def _copy_placeholder_meta(
    placeholder: fx.Node, input_node: fx.Node, owning_module: fx.GraphModule
) -> None:
    if "val" in input_node.meta:
        placeholder.meta.update(input_node.meta)
    elif input_node.op == "get_attr" and isinstance(input_node.target, str):
        placeholder.meta["val"] = attrgetter(input_node.target)(owning_module)


def _register_subgraph(
    graph: fx.Graph,
    subgraph_module: fx.GraphModule,
    name: str,
) -> str:
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("expected graph to have an owning_module")
    subgraph_attr_name = get_subgraph_name(owning_module, name)
    setattr(owning_module, subgraph_attr_name, subgraph_module)
    return subgraph_attr_name


def _insert_control_deps_call(
    graph: fx.Graph,
    get_subgraph: fx.Node,
    additional_deps: tuple[fx.Node, ...] | list[fx.Node],
    args: tuple[fx.Node, ...] | list[fx.Node],
    kwargs: dict[str, Any] | None = None,
    name: str | None = None,
) -> fx.Node:
    with graph.inserting_after(get_subgraph):
        return graph.call_function(
            control_deps,
            args=(tuple(additional_deps), get_subgraph, *tuple(args)),
            kwargs=kwargs or {},
            name=name,
        )


def _extract_unique_nodes(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[fx.Node], list[Any], Any]:
    """Extract unique fx.Node instances from args/kwargs using pytree.

    Args:
        args: The positional arguments (may contain nested structures with fx.Node)
        kwargs: The keyword arguments (may contain nested structures with fx.Node)

    Returns:
        - Ordered list of unique fx.Node instances (preserves first occurrence order)
        - Flattened list of all items from args/kwargs
        - The pytree spec for reconstructing the original structure
    """
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))
    unique_nodes: list[fx.Node] = []
    seen: OrderedSet[fx.Node] = OrderedSet()
    for item in flat_args_kwargs:
        if isinstance(item, fx.Node) and item not in seen:
            unique_nodes.append(item)
            seen.add(item)
    return unique_nodes, flat_args_kwargs, spec


def mark_fuse_region(
    graph: fx.Graph,
    nodes: list[fx.Node],
) -> fx.Node | tuple[fx.Node, ...]:
    """
    Wrap a region of FX nodes with control_deps(..., fuse_region=True).

    This is the pass-facing helper for introducing a fusion barrier around an
    existing set of FX nodes.  The selected nodes are outlined into a subgraph
    and replaced by a control_deps HOP call with an empty ``additional_deps``
    tuple and ``fuse_region=True``.

    Semantics:
    - Inductor may fuse operations produced by this subgraph with each other.
    - Those operations may not fuse with operations outside this region.
    - Separately marked regions get distinct region ids and may not fuse with
      each other.
    - Operations before and after the region are still outside the region and
      may fuse with each other if the normal scheduler rules allow it.
    - This does not add an execution-ordering dependency by itself; use
      ``additional_deps``/``preserve_node_ordering`` when ordering is required.
    - A region with no external consumers is allowed.  It outlines to a
      subgraph returning ``()`` and leaves the control_deps node in place.

    The helper preserves the original graph boundary: external producer nodes
    become HOP operands, and nodes used outside the region become replacement
    outputs.  HOP subgraph operands stored as GraphModule ``get_attr`` nodes are
    kept as ``get_attr`` inside the outlined graph so recursive subgraph passes
    can still discover them.
    """
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("expected graph to have an owning_module")
    if not nodes:
        raise AssertionError("expected non-empty nodes")

    node_set = OrderedSet(nodes)
    ordered_nodes = [node for node in graph.nodes if node in node_set]
    if len(ordered_nodes) != len(node_set):
        raise AssertionError("expected all nodes to belong to graph")

    region_outputs = [
        node
        for node in ordered_nodes
        if any(user not in node_set for user in node.users)
    ]

    subgraph = fx.Graph(owning_module)
    env: dict[fx.Node, fx.Node] = {}
    placeholders: dict[fx.Node, fx.Node] = {}

    external_inputs: OrderedSet[fx.Node] = OrderedSet()
    preserved_getattrs: OrderedSet[fx.Node] = OrderedSet()

    def collect_external_input(node: fx.Node) -> fx.Node:
        if node not in node_set:
            # HOP subgraph operands must stay as get_attr nodes so recursive
            # subgraph passes can still find them after outlining.
            if (
                node.op == "get_attr"
                and isinstance(node.target, str)
                and isinstance(attrgetter(node.target)(owning_module), fx.GraphModule)
            ):
                preserved_getattrs.add(node)
            else:
                external_inputs.add(node)
        return node

    for node in ordered_nodes:
        fx.map_arg((node.args, node.kwargs), collect_external_input)

    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    latest_input = max(
        external_inputs,
        key=lambda node: node_order[node],
        default=None,
    )
    first_external_user = min(
        (
            user
            for output_node in region_outputs
            for user in output_node.users
            if user not in node_set
        ),
        key=lambda node: node_order[node],
        default=None,
    )
    if (
        first_external_user is not None
        and latest_input is not None
        and node_order[latest_input] >= node_order[first_external_user]
    ):
        raise AssertionError("expected fuse_region boundary to be acyclic")

    for input_node in external_inputs:
        placeholder = subgraph.placeholder(f"arg_{len(placeholders)}")
        _copy_placeholder_meta(placeholder, input_node, owning_module)
        placeholders[input_node] = placeholder

    def load_arg(node: fx.Node) -> fx.Node:
        if node in env:
            return env[node]
        if node in node_set:
            raise AssertionError("expected fuse_region nodes to be topological")
        if node in preserved_getattrs:
            if not isinstance(node.target, str):
                raise AssertionError("expected get_attr target to be a string")
            get_attr_node = subgraph.get_attr(node.target)
            get_attr_node.meta.update(node.meta)
            env[node] = get_attr_node
            return get_attr_node
        return placeholders[node]

    for node in ordered_nodes:
        env[node] = subgraph.node_copy(node, load_arg)

    subgraph_outputs = tuple(env[node] for node in region_outputs)
    if len(subgraph_outputs) == 0:
        out = subgraph.output(())
        out.meta["val"] = ()
    elif len(subgraph_outputs) == 1:
        out = subgraph.output(subgraph_outputs[0])
        if "val" in region_outputs[0].meta:
            out.meta["val"] = region_outputs[0].meta["val"]
    else:
        out = subgraph.output(subgraph_outputs)
        out.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
    subgraph.lint()

    subgraph_module = _LazyGraphModule(owning_module, subgraph)
    region_name = f"fuse_region_{ordered_nodes[0].name}_{ordered_nodes[-1].name}"
    subgraph_attr_name = _register_subgraph(graph, subgraph_module, region_name)

    if latest_input is None or node_order[latest_input] < node_order[ordered_nodes[0]]:
        with graph.inserting_before(ordered_nodes[0]):
            get_subgraph = graph.get_attr(subgraph_attr_name)
    else:
        with graph.inserting_after(latest_input):
            get_subgraph = graph.get_attr(subgraph_attr_name)

    region_node = _insert_control_deps_call(
        graph,
        get_subgraph,
        (),
        tuple(placeholders),
        kwargs={FUSE_REGION: True},
        name=region_name,
    )

    replacements: list[fx.Node] = []
    if len(region_outputs) == 0:
        region_node.meta["val"] = ()
    elif len(region_outputs) == 1:
        replacement = region_node
        replacement.meta = region_outputs[0].meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        replacements.append(replacement)
    else:
        region_node.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
        insert_after = region_node
        for idx, output_node in enumerate(region_outputs):
            with graph.inserting_after(insert_after):
                replacement = graph.call_function(
                    getitem,
                    args=(region_node, idx),
                    name=f"{output_node.name}_fuse_region",
                )
            replacement.meta = output_node.meta.copy()
            replacement.meta.pop("eager_input_vals", None)
            replacements.append(replacement)
            insert_after = replacement

    for output_node, replacement in zip(region_outputs, replacements, strict=True):
        for user in list(output_node.users):
            if user not in node_set:
                user.replace_input_with(output_node, replacement)

    for node in reversed(ordered_nodes):
        graph.erase_node(node)
    graph.lint()

    if not replacements:
        return region_node
    return replacements[0] if len(replacements) == 1 else tuple(replacements)


def preserve_node_ordering(
    graph: fx.Graph,
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    verbose: bool = False,
) -> None:
    """
    Preserve node ordering using control_deps HOP with subgraph.

    This function wraps operations with control_deps that:
    1. Makes additional dependencies explicit (first argument)
    2. Creates a subgraph internally to preserve the exact original operation
    3. Preserves the original node names

    Args:
        graph: The FX graph to modify
        additional_deps_map: Mapping from dependent nodes to their dependencies
        verbose: If True, print debug information
    """
    if not additional_deps_map:
        return

    # Track replacements so we can update dependencies
    replacements: dict[fx.Node, fx.Node] = {}

    # Process each node that needs additional dependencies
    for dependent_node, dep_nodes in additional_deps_map.items():
        if dependent_node.op != "call_function":
            raise AssertionError(dependent_node.op)

        original_name = dependent_node.name
        original_args = dependent_node.args
        original_kwargs = dependent_node.kwargs
        original_meta = dependent_node.meta.copy()

        updated_dep_nodes = [replacements.get(dep, dep) for dep in dep_nodes]

        # Create a subgraph that preserves the exact original operation
        subgraph_module = _create_subgraph_for_node(graph, dependent_node)

        subgraph_attr_name = _register_subgraph(graph, subgraph_module, original_name)

        # Create control_deps call with:
        # 1. Additional dependencies as first arg (explicit)
        # 2. Subgraph via get_attr (like b2b gemm pass)
        # 3. Original arguments (only fx.Node args and kwargs are passed)
        with graph.inserting_before(dependent_node):
            # Create get_attr node for the subgraph
            get_subgraph = graph.get_attr(subgraph_attr_name)

            # Extract unique nodes from nested args/kwargs
            node_args, _, _ = _extract_unique_nodes(original_args, original_kwargs)

        # Create with temporary name first to avoid conflict with the original
        # node, which is still in the graph.
        ordered_node = _insert_control_deps_call(
            graph,
            get_subgraph,
            updated_dep_nodes,
            node_args,
            name=f"__temp_{original_name}",
        )

        # Copy metadata from original node
        ordered_node.meta = original_meta
        # this will be constrained on the target node in subgraph if it exists
        ordered_node.meta.pop("eager_input_vals", None)

        # Replace all uses of the original node with the ordered version
        dependent_node.replace_all_uses_with(ordered_node)

        # Remove the original node from the graph
        graph.erase_node(dependent_node)

        # Now rename the ordered node to the original name
        ordered_node.name = original_name  # PRESERVE ORIGINAL NAME

        # Track the replacement for future dependencies
        replacements[dependent_node] = ordered_node


def _create_subgraph_for_node(
    graph: fx.Graph, node: fx.Node, additional_deps=None
) -> fx.GraphModule:
    """
    Create a subgraph that exactly recreates a node's operation optionally passing through additional dependencies.

    The subgraph takes only the fx.Node arguments and recreates the operation
    with the exact target, args structure, and kwargs.

    Args:
        graph: The parent graph
        node: The node to wrap in a subgraph
        additional_deps: Additional dependencies to pass through the subgraph

    Returns:
        A GraphModule containing the subgraph
    """
    # Get the owning module
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("graph.owning_module must not be None")

    # Create a new graph for the subgraph
    subgraph = fx.Graph(owning_module)

    # Extract unique nodes and get flattened structure + spec
    unique_nodes, flat_args_kwargs, spec = _extract_unique_nodes(node.args, node.kwargs)

    # Create placeholders for each unique node
    node_to_placeholder: dict[fx.Node, fx.Node] = {}
    for idx, orig_node in enumerate(unique_nodes):
        placeholder = subgraph.placeholder(f"arg_{idx}")
        _copy_placeholder_meta(placeholder, orig_node, owning_module)
        node_to_placeholder[orig_node] = placeholder

    # Replace fx.Node instances with their placeholders
    def replace_nodes(item: Any) -> Any:
        if isinstance(item, fx.Node):
            return node_to_placeholder[item]
        return item

    additional_deps_placeholders = []
    for idx, dep in enumerate(additional_deps or ()):
        placeholder = subgraph.placeholder(f"dep_{idx}")
        _copy_placeholder_meta(placeholder, dep, owning_module)
        additional_deps_placeholders.append(placeholder)

    new_flat = [replace_nodes(item) for item in flat_args_kwargs]
    new_args, new_kwargs = pytree.tree_unflatten(new_flat, spec)

    # Recreate the exact original operation in the subgraph
    if not callable(node.target):
        raise AssertionError(f"expected node.target to be callable, got {node.target}")
    result = subgraph.call_function(
        node.target,
        tuple(new_args),
        new_kwargs,  # type: ignore[arg-type]
    )

    # Copy metadata from the original node
    result.meta.update(node.meta)

    if additional_deps_placeholders:
        outputs = tuple([result] + additional_deps_placeholders)
        out = subgraph.output(outputs)
        out.meta["val"] = tuple(output.meta.get("val") for output in outputs)
    else:
        out = subgraph.output(result)
        if "val" in result.meta:
            out.meta["val"] = result.meta["val"]

    return _LazyGraphModule(owning_module, subgraph)
