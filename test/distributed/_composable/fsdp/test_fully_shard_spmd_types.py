# Owner(s): ["oncall: distributed"]

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    ShardPlacementResult,
)
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info
from torch.distributed.tensor import init_device_mesh, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


if dist._is_spmd_types_available():
    import spmd_types as spmd
    from spmd_types.checker import typecheck


class SpmdLinear(nn.Module):
    def __init__(self, mesh, seq_parallel: bool):
        super().__init__()
        self.mesh = mesh
        self.tp_pg = mesh.get_group("tp")
        self.seq_parallel = seq_parallel
        self.unsharded_weight = nn.Parameter(torch.randn(16, 16))
        self.sharded_weight = nn.Parameter(torch.randn(16, 16))
        self.compute_param_types = None

    def forward(self, x):
        """Simulate the TP collectives around a sharded projection.

        The global computation is x = x @ A; x = x @ B; return x.sum().
        With sequence parallelism, the activation is all-gathered before the
        sharded projection. Without it, the output is all-gathered before loss.
        """
        self.compute_param_types = (
            dict(spmd.get_local_type(self.unsharded_weight)),
            dict(spmd.get_local_type(self.sharded_weight)),
        )
        x = x @ self.unsharded_weight
        x = spmd.redistribute(
            x,
            self.tp_pg,
            src=spmd.S(1) if self.seq_parallel else spmd.I,
            dst=spmd.R,
            backward_options={"op_dtype": torch.float32},
        )
        x = x @ self.sharded_weight.t()
        x = spmd.redistribute(
            x,
            self.tp_pg,
            src=spmd.S(2),
            dst=spmd.I,
            backward_options={"op_dtype": torch.float32},
        )
        return x.sum()


class SpmdParamOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 16))
        self.compute_param_type = None

    def forward(self):
        self.compute_param_type = dict(spmd.get_local_type(self.weight))
        return self.weight.sum()


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestFullyShardSpmdTypes(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=0, world_size=16
        )
        cls.type_mesh = init_device_mesh(
            "cpu", (4, 2, 2), mesh_dim_names=("dp", "cp", "tp")
        )
        cls.fsdp_mesh = init_device_mesh(
            "cpu", (2, 2, 2, 2), mesh_dim_names=("dpr", "dps", "cp", "tp")
        )
        cls.sparse_mesh = init_device_mesh(
            "cpu", (4, 4), mesh_dim_names=("efsdp", "ep")
        )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDownClass()

    def test_restores_param_spmd_type_for_compute(self):
        """FSDP restores user SPMD metadata on params for compute.

        FSDP should preserve the compute-mesh annotations when applied with
        a different storage-mesh view.
        """
        dp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("dp"))
        cp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("cp"))
        tp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("tp"))
        for seq_parallel in (False, True):
            with self.subTest(seq_parallel=seq_parallel):
                model = SpmdLinear(self.type_mesh, seq_parallel)
                spmd.assert_type(
                    model.unsharded_weight,
                    {
                        dp_axis: spmd.R,
                        cp_axis: spmd.R,
                        tp_axis: spmd.R if seq_parallel else spmd.I,
                    },
                )
                spmd.assert_type(
                    model.sharded_weight,
                    {dp_axis: spmd.R, cp_axis: spmd.R, tp_axis: spmd.S(0)},
                )

                # FSDP turns params into DTensor, in this case should contain StridedShard @ dps
                with spmd.set_current_mesh(self.type_mesh):
                    fully_shard(
                        model,
                        mesh=self.fsdp_mesh,
                        dp_mesh_dims=DataParallelMeshDims(
                            shard=("dps", "cp"),
                            replicate="dpr",
                        ),
                        mp_policy=MixedPrecisionPolicy(
                            param_dtype=torch.bfloat16,
                            reduce_dtype=torch.float32,
                            cast_forward_inputs=True,
                        ),
                    )
                self.assertEqual(
                    model.sharded_weight._spec.placements,
                    (
                        Replicate(),
                        _StridedShard(0, split_factor=self.type_mesh["tp"].size()),
                        Shard(0),
                    ),
                )

                # annotate model inputs as V + PartitionSpec
                inp = torch.randn(4, 8, 16)
                input_type = {
                    dp_axis: spmd.V,
                    cp_axis: spmd.V,
                    tp_axis: spmd.V if seq_parallel else spmd.I,
                }
                input_partition_spec = spmd.PartitionSpec(
                    dp_axis,
                    (cp_axis, tp_axis) if seq_parallel else cp_axis,
                    None,
                )

                # check loss output type, check compute-time param annotations are restored.
                with (
                    spmd.set_current_mesh(self.type_mesh),
                    typecheck(strict_mode="strict", local=False),
                ):
                    spmd.assert_type(
                        inp,
                        input_type,
                        partition_spec=input_partition_spec,
                    )
                    loss = model(inp)
                    spmd.assert_type(
                        loss,
                        {dp_axis: spmd.P, cp_axis: spmd.P, tp_axis: spmd.I},
                    )
                self.assertEqual(
                    model.compute_param_types,
                    (
                        {
                            dp_axis: spmd.R,
                            cp_axis: spmd.R,
                            tp_axis: spmd.R if seq_parallel else spmd.I,
                        },
                        {dp_axis: spmd.R, cp_axis: spmd.R, tp_axis: spmd.V},
                    ),
                )

                with CommDebugMode() as comm_mode:
                    loss.backward()
                comm_counts = comm_mode.get_comm_counts()
                if seq_parallel:
                    self.assertEqual(
                        comm_counts[torch.ops.c10d._reduce_scatter_base_], 2
                    )
                else:
                    self.assertEqual(
                        comm_counts[torch.ops.c10d._reduce_scatter_base_], 1
                    )
                self.assertEqual(
                    comm_counts[torch.ops.c10d.allreduce_]
                    + comm_counts[torch.ops.c10d_functional.all_reduce],
                    2,
                )

    def test_local_v_param_requires_partition_spec(self):
        """Local-only V@TP params are ambiguous for FSDP.

        Without PartitionSpec shard info, FSDP cannot choose the matching
        DTensor Shard(dim), so it rejects the parameter at init time.
        """
        model = SpmdLinear(self.type_mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.type_mesh):
            spmd.assert_type(
                model.unsharded_weight,
                {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
            )
            spmd.assert_type(
                model.sharded_weight,
                {"dp": spmd.R, "cp": spmd.R, "tp": spmd.V},
            )
        with self.assertRaises(ValueError) as cm, spmd.set_current_mesh(self.type_mesh):
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            """Cannot convert plain Varying to a DTensor placement. Use S(dim) to specify which tensor dimension is sharded.""",
        )

    def test_partial_param_annotations_infer_fsdp_axes_at_compute(self):
        model = SpmdLinear(self.type_mesh, seq_parallel=False)
        dp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("dp"))
        cp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("cp"))
        tp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("tp"))

        spmd.assert_type(
            model.unsharded_weight,
            {tp_axis: spmd.I},
        )
        spmd.assert_type(
            model.sharded_weight,
            {tp_axis: spmd.S(0)},
        )
        with spmd.set_current_mesh(self.type_mesh):
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
            )

        inp = torch.randn(4, 8, 16)
        with (
            spmd.set_current_mesh(self.type_mesh),
            typecheck(strict_mode="strict", local=False),
        ):
            spmd.assert_type(
                inp,
                {dp_axis: spmd.V, cp_axis: spmd.V, tp_axis: spmd.I},
                partition_spec=spmd.PartitionSpec(dp_axis, cp_axis, None),
            )
            loss = model(inp)
            spmd.assert_type(
                loss,
                {dp_axis: spmd.P, cp_axis: spmd.P, tp_axis: spmd.I},
            )
        self.assertEqual(
            model.compute_param_types,
            (
                {dp_axis: spmd.R, cp_axis: spmd.R, tp_axis: spmd.I},
                {dp_axis: spmd.R, cp_axis: spmd.R, tp_axis: spmd.V},
            ),
        )

    def test_shard_placement_result_spmd_compute_mesh_infers_fsdp_axes(self):
        model = SpmdLinear(self.type_mesh, seq_parallel=False)
        tp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("tp"))
        spmd.assert_type(model.unsharded_weight, {tp_axis: spmd.I})
        spmd.assert_type(model.sharded_weight, {tp_axis: spmd.S(0)})

        mesh_info = _get_mesh_info(
            self.fsdp_mesh,
            DataParallelMeshDims(shard=("dps", "cp"), replicate="dpr"),
        )
        self.assertIsInstance(mesh_info, FSDPMeshInfo)

        def shard_placement_fn(param):
            return ShardPlacementResult(
                placement=Shard(0),
                mesh_info=mesh_info,
                spmd_compute_mesh=self.type_mesh,
            )

        fully_shard(
            model,
            mesh=self.fsdp_mesh,
            dp_mesh_dims=DataParallelMeshDims(
                shard=("dps", "cp"),
                replicate="dpr",
            ),
            shard_placement_fn=shard_placement_fn,
        )

        dp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("dp"))
        cp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("cp"))
        inp = torch.randn(4, 8, 16)
        with (
            spmd.set_current_mesh(self.type_mesh),
            typecheck(strict_mode="strict", local=False),
        ):
            spmd.assert_type(
                inp,
                {dp_axis: spmd.V, cp_axis: spmd.V, tp_axis: spmd.I},
                partition_spec=spmd.PartitionSpec(dp_axis, cp_axis, None),
            )
            loss = model(inp)
            spmd.assert_type(
                loss,
                {dp_axis: spmd.P, cp_axis: spmd.P, tp_axis: spmd.I},
            )
        self.assertEqual(
            model.compute_param_types,
            (
                {dp_axis: spmd.R, cp_axis: spmd.R, tp_axis: spmd.I},
                {dp_axis: spmd.R, cp_axis: spmd.R, tp_axis: spmd.V},
            ),
        )

    def test_spmd_compute_mesh_axes_must_match_annotations(self):
        model = SpmdParamOnly()
        tp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("tp"))
        spmd.assert_type(model.weight, {tp_axis: spmd.I})

        mesh_info = _get_mesh_info(
            self.fsdp_mesh,
            DataParallelMeshDims(shard=("dps", "cp"), replicate="dpr"),
        )
        self.assertIsInstance(mesh_info, FSDPMeshInfo)

        def shard_placement_fn(param):
            return ShardPlacementResult(
                placement=Shard(0),
                mesh_info=mesh_info,
                spmd_compute_mesh=self.sparse_mesh,
            )

        with self.assertRaises(ValueError) as cm:
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
                shard_placement_fn=shard_placement_fn,
            )
        self.assertIn(
            "annotations on axes that are not in the resolved typechecking mesh",
            str(cm.exception),
        )

    def test_spmd_compute_mesh_must_span_storage_mesh(self):
        model = SpmdLinear(self.type_mesh, seq_parallel=False)
        tp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("tp"))
        spmd.assert_type(model.unsharded_weight, {tp_axis: spmd.I})

        mesh_info = _get_mesh_info(
            self.fsdp_mesh,
            DataParallelMeshDims(shard=("dps", "cp"), replicate="dpr"),
        )
        self.assertIsInstance(mesh_info, FSDPMeshInfo)

        def shard_placement_fn(param):
            return ShardPlacementResult(
                placement=Shard(0),
                mesh_info=mesh_info,
                spmd_compute_mesh=self.type_mesh["tp"],
            )

        with self.assertRaises(ValueError) as cm:
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
                shard_placement_fn=shard_placement_fn,
            )
        self.assertExpectedInline(
            str(cm.exception),
            "Parameter 'unsharded_weight' uses spmd_types annotations on a "
            "typechecking mesh that does not span the same ranks as its FSDP "
            "storage mesh. FSDP can fill omitted FSDP-managed axes only when "
            "these meshes span the same rank set. Typechecking mesh axes: "
            "(mesh_tp,). Storage mesh axes: (mesh_dpr, mesh_dps, mesh_cp, "
            "mesh_tp).",
        )

    def test_spmd_compute_mesh_same_size_different_span_errors(self):
        model = SpmdParamOnly()
        compute_mesh = DeviceMesh(
            "cpu",
            torch.arange(0, 16, 2),
            mesh_dim_names=("even",),
        )
        compute_axis = spmd.MeshAxis.of(compute_mesh.get_group("even"))
        spmd.assert_type(model.weight, {compute_axis: spmd.I})

        storage_mesh = DeviceMesh(
            "cpu",
            torch.arange(8).reshape(2, 2, 2),
            mesh_dim_names=("half_dpr", "half_dps", "half_tp"),
        )
        mesh_info = _get_mesh_info(
            storage_mesh,
            DataParallelMeshDims(shard="half_dps", replicate="half_dpr"),
        )
        self.assertIsInstance(mesh_info, FSDPMeshInfo)

        def shard_placement_fn(param):
            return ShardPlacementResult(
                placement=Shard(0),
                mesh_info=mesh_info,
                spmd_compute_mesh=compute_mesh,
            )

        with self.assertRaises(ValueError) as cm:
            fully_shard(
                model,
                mesh=storage_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard="half_dps",
                    replicate="half_dpr",
                ),
                shard_placement_fn=shard_placement_fn,
            )
        self.assertIn(
            "FSDP can fill omitted FSDP-managed axes only when these meshes "
            "span the same rank set.",
            str(cm.exception),
        )

    def test_shard_placement_fn_defaults_compute_mesh_to_storage_mesh(self):
        model = SpmdParamOnly()
        efsdp_axis = spmd.MeshAxis.of(self.sparse_mesh.get_group("efsdp"))
        ep_axis = spmd.MeshAxis.of(self.sparse_mesh.get_group("ep"))
        spmd.assert_type(model.weight, {ep_axis: spmd.S(0)})

        mesh_info = _get_mesh_info(
            self.sparse_mesh,
            DataParallelMeshDims(shard="efsdp"),
        )
        self.assertIsInstance(mesh_info, FSDPMeshInfo)

        def shard_placement_fn(param):
            return ShardPlacementResult(
                placement=Shard(0),
                mesh_info=mesh_info,
            )

        fully_shard(
            model,
            mesh=self.sparse_mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="efsdp"),
            shard_placement_fn=shard_placement_fn,
        )

        with (
            spmd.set_current_mesh(self.sparse_mesh),
            typecheck(strict_mode="strict", local=False),
        ):
            out = model()
            spmd.assert_type(out, {efsdp_axis: spmd.R, ep_axis: spmd.P})
        self.assertEqual(
            model.compute_param_type,
            {efsdp_axis: spmd.R, ep_axis: spmd.V},
        )

    def test_spmd_params_require_dp_mesh_dims(self):
        model = SpmdLinear(self.type_mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.type_mesh):
            spmd.assert_type(
                model.unsharded_weight,
                {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
            )
        with self.assertRaises(ValueError) as cm, spmd.set_current_mesh(self.type_mesh):
            fully_shard(model, mesh=self.type_mesh["dp"])
        self.assertExpectedInline(
            str(cm.exception),
            "spmd_types parameters require a named SPMD mesh "
            "(pass dp_mesh_dims to fully_shard)",
        )

    def test_partial_param_annotations_require_init_compute_mesh(self):
        model = SpmdLinear(self.type_mesh, seq_parallel=False)
        tp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("tp"))
        spmd.assert_type(model.unsharded_weight, {tp_axis: spmd.I})

        with self.assertRaises(ValueError) as cm:
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            "spmd_types parameters require an init-time compute mesh for FSDP "
            "annotation restore. If all parameters in the FSDP unit share a "
            "typechecking mesh, wrap fully_shard() in spmd.set_current_mesh(). "
            "For mixed parameter groups, set DataParallelMeshInfo.spmd_mesh "
            "or FSDPMeshInfo.spmd_mesh when the typechecking mesh matches the "
            "FSDP storage mesh, or set ShardPlacementResult.spmd_compute_mesh "
            "when the typechecking and storage meshes differ.",
        )

    def test_partial_param_annotations_missing_non_fsdp_axis_errors(self):
        model = SpmdLinear(self.type_mesh, seq_parallel=False)
        dp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("dp"))
        cp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("cp"))
        tp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("tp"))

        spmd.assert_type(
            model.unsharded_weight,
            {dp_axis: spmd.R, cp_axis: spmd.R},
        )
        spmd.assert_type(
            model.sharded_weight,
            {tp_axis: spmd.S(0)},
        )
        with self.assertRaises(ValueError) as cm, spmd.set_current_mesh(self.type_mesh):
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            "Parameter 'unsharded_weight' has incomplete spmd_types annotations "
            "for FSDP storage. Annotated axes: (mesh_dp, mesh_cp). Storage "
            "mesh axes: (mesh_dpr, mesh_dps, mesh_cp, mesh_tp). FSDP mesh "
            "dims: DataParallelMeshDims(shard=('dps', 'cp'), "
            "replicate='dpr'). Missing non-FSDP storage axes: "
            "(mesh_tp,).",
        )

    def test_fsdp_dp_axes_must_be_r(self):
        model = SpmdLinear(self.type_mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.type_mesh):
            spmd.assert_type(
                model.unsharded_weight,
                {"dp": spmd.I, "cp": spmd.R, "tp": spmd.I},
            )
        with self.assertRaises(ValueError) as cm, spmd.set_current_mesh(self.type_mesh):
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            "Expected spmd.R on FSDP DP axis mesh_dp for parameter "
            "'unsharded_weight' but got PerMeshAxisLocalSpmdType.I. FSDP "
            "requires DP parameters to be R since it handles the DP gradient "
            "reduction.",
        )


if __name__ == "__main__":
    run_tests()
