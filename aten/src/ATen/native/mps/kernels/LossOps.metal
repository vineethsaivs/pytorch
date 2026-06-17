// LossOps.metal
// Metal compute kernels replacing MPSGraph ops in LossOps.mm.
// Eliminates shape-dependent graph cache growth for all loss operations.
//
// Reduction codes match ATen: 0=None, 1=Mean, 2=Sum
// Kernel naming convention:
//   <op>_fwd_none_<T>  : None reduction, writes typed T* output
//   <op>_fwd_reduce_<T>: Mean/Sum, writes float* partial sums (one per TG)
//   <op>_bwd_<T>       : backward, grad_out scalar (reduce) or per-elem (none)

#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <metal_stdlib>
using namespace metal;

// ──────────────────────────────────────────────────────────────────────────
// Param structs  (binary layout must match C++ structs in LossOps.mm)
// ──────────────────────────────────────────────────────────────────────────

struct BCEParams {
  uint32_t N;
  float scale;
  uint32_t reduction;
  uint32_t has_weight;
};

// ──────────────────────────────────────────────────────────────────────────
// Threadgroup tree-reduce helper
// ──────────────────────────────────────────────────────────────────────────

template <typename T>
inline T tg_sum(T val, threadgroup T* smem, uint lid, uint tgsz) {
  // SIMD-group reduce: hardware, zero barriers within warp
  float f = simd_sum(float(val));
  if ((lid & 31u) == 0u)
    smem[lid >> 5u] = T(f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    float acc = 0.f;
    for (uint g = 0u; g < (tgsz >> 5u); ++g)
      acc += float(smem[g]);
    smem[0] = T(acc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return smem[0];
}

// ──────────────────────────────────────────────────────────────────────────
// Phase-2: merge per-threadgroup float partials into loss[0]
// Always dispatched with a single 256-thread threadgroup.
// ──────────────────────────────────────────────────────────────────────────

template <typename T>
kernel void loss_reduce_partials_typed(
    device const float* partial [[buffer(0)]],
    device T* loss [[buffer(1)]],
    constant uint32_t& nparts [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]]) {
  threadgroup float smem[256];
  float acc = 0.f;
  for (uint i = lid; i < nparts; i += tgsz)
    acc += partial[i];
  acc = tg_sum(acc, smem, lid, tgsz);
  if (lid == 0)
    loss[0] = T(acc);
}

#define INST_REDUCE_PARTIALS(T)                      \
  template [[host_name("loss_reduce_partials_" #T)]] \
  kernel void loss_reduce_partials_typed<T>(         \
      device const float*, device T*, constant uint32_t&, uint, uint)
INST_REDUCE_PARTIALS(float);
INST_REDUCE_PARTIALS(half);
INST_REDUCE_PARTIALS(bfloat);

// ============================================================================
// BCE Loss  (binary cross-entropy; input clamped to (eps, 1-eps))
// ============================================================================

template <typename T>
kernel void bce_loss_fwd_none(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant BCEParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  auto bce_elem = [&](float x_raw, float y, float w) -> float {
    float x = clamp(x_raw, 1e-7f, 1.f - 1e-7f);
    float l = -y * log(x) - (1.f - y) * log(1.f - x);
    return p.has_weight ? l * w : l;
  };
  uint base = gid * 4u;
  if (base + 4u <= p.N) {
    using T4 = vec<T, 4>;
    T4 i4 = reinterpret_cast<device const T4*>(input)[gid];
    T4 t4 = reinterpret_cast<device const T4*>(target)[gid];
    T4 w4 = reinterpret_cast<device const T4*>(weight)[gid];
    T4 res;
    res[0] = T(bce_elem(float(i4[0]), float(t4[0]), float(w4[0])));
    res[1] = T(bce_elem(float(i4[1]), float(t4[1]), float(w4[1])));
    res[2] = T(bce_elem(float(i4[2]), float(t4[2]), float(w4[2])));
    res[3] = T(bce_elem(float(i4[3]), float(t4[3]), float(w4[3])));
    reinterpret_cast<device T4*>(out)[gid] = res;
  } else {
    for (uint i = base; i < p.N; i++) {
      float x = clamp(float(input[i]), 1e-7f, 1.f - 1e-7f);
      float y = float(target[i]);
      float w = p.has_weight ? float(weight[i]) : 1.f;
      out[i] = T(bce_elem(x, y, w));
    }
  }
}

template <typename T>
kernel void bce_loss_fwd_reduce(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device float* partial [[buffer(3)]],
    constant BCEParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  threadgroup float smem[256];
  float acc = 0.f;
  using T4 = vec<T, 4>;
  auto bce_elem = [&](float x_in, float y) -> float {
    float x = clamp(x_in, 1e-7f, 1.f - 1e-7f);
    return -y * log(x) - (1.f - y) * log(1.f - x);
  };
  uint aligned_N = (p.N / 4u) * 4u;
  uint base = gid * 4u;
  uint vec_stride = tpg * 4u;
  while (base + 4u <= aligned_N) {
    T4 i4 = reinterpret_cast<device const T4*>(input)[base / 4u];
    T4 t4 = reinterpret_cast<device const T4*>(target)[base / 4u];
    float4 x4 = float4(i4);
    float4 y4 = float4(t4);
    float4 l4 = float4(
        bce_elem(x4.x, y4.x),
        bce_elem(x4.y, y4.y),
        bce_elem(x4.z, y4.z),
        bce_elem(x4.w, y4.w));
    if (p.has_weight) {
      T4 w4 = reinterpret_cast<device const T4*>(weight)[base / 4u];
      float4 wf4 = float4(w4);
      l4 *= wf4;
    }
    acc += l4.x + l4.y + l4.z + l4.w;
    base += vec_stride;
  }
  for (uint i = aligned_N + gid; i < p.N; i += tpg) {
    float x = clamp(float(input[i]), 1e-7f, 1.f - 1e-7f);
    float y = float(target[i]);
    float l = -y * log(x) - (1.f - y) * log(1.f - x);
    acc += p.has_weight ? l * float(weight[i]) : l;
  }
  acc = tg_sum(acc, smem, lid, tgsz);
  if (lid == 0)
    partial[tgid] = acc * p.scale;
}

// REVERTED to v2 (the high-water mark): single kernel, 16-elem hand-unrolled
// batched-loads, runtime if (p.reduction == 0) branch. Each subsequent
// iteration (v3 32-elem unified, v4/v5 split, v6 vec<T,8>) was a net
// regression vs this form. Further changes will be informed by Metal frame
// capture data, not hypothesis.
template <typename T>
kernel void bce_loss_bwd(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* target [[buffer(2)]],
    device const T* weight [[buffer(3)]],
    device T* grad_in [[buffer(4)]],
    constant BCEParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  using T4 = vec<T, 4>;
  auto dx_vec4 = [](float4 x_raw, float4 y) -> float4 {
    // Match CPU semantics: numerator uses raw x (so grad=0 at x=y=0 or x=y=1),
    // denominator uses clamped x for numerical stability.
    float4 x = clamp(x_raw, 1e-7f, 1.f - 1e-7f);
    return (x_raw - y) / (x * (1.f - x));
  };
  float g_scalar = (p.reduction != 0) ? float(grad_out[0]) * p.scale : 0.f;
  uint base = gid * 16u;
  if (base + 16u <= p.N) {
    uint t = gid * 4u;
    T4 i0 = reinterpret_cast<device const T4*>(input)[t + 0u];
    T4 i1 = reinterpret_cast<device const T4*>(input)[t + 1u];
    T4 i2 = reinterpret_cast<device const T4*>(input)[t + 2u];
    T4 i3 = reinterpret_cast<device const T4*>(input)[t + 3u];
    T4 a0 = reinterpret_cast<device const T4*>(target)[t + 0u];
    T4 a1 = reinterpret_cast<device const T4*>(target)[t + 1u];
    T4 a2 = reinterpret_cast<device const T4*>(target)[t + 2u];
    T4 a3 = reinterpret_cast<device const T4*>(target)[t + 3u];
    float4 dx0 = dx_vec4(float4(i0), float4(a0));
    float4 dx1 = dx_vec4(float4(i1), float4(a1));
    float4 dx2 = dx_vec4(float4(i2), float4(a2));
    float4 dx3 = dx_vec4(float4(i3), float4(a3));
    if (p.has_weight) {
      dx0 *= float4(reinterpret_cast<device const T4*>(weight)[t + 0u]);
      dx1 *= float4(reinterpret_cast<device const T4*>(weight)[t + 1u]);
      dx2 *= float4(reinterpret_cast<device const T4*>(weight)[t + 2u]);
      dx3 *= float4(reinterpret_cast<device const T4*>(weight)[t + 3u]);
    }
    float4 g0, g1, g2, g3;
    if (p.reduction == 0) {
      g0 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 0u]) * dx0;
      g1 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 1u]) * dx1;
      g2 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 2u]) * dx2;
      g3 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 3u]) * dx3;
    } else {
      g0 = g_scalar * dx0;
      g1 = g_scalar * dx1;
      g2 = g_scalar * dx2;
      g3 = g_scalar * dx3;
    }
    reinterpret_cast<device T4*>(grad_in)[t + 0u] = T4(g0);
    reinterpret_cast<device T4*>(grad_in)[t + 1u] = T4(g1);
    reinterpret_cast<device T4*>(grad_in)[t + 2u] = T4(g2);
    reinterpret_cast<device T4*>(grad_in)[t + 3u] = T4(g3);
  } else {
    for (uint i = base; i < p.N; i++) {
      float x_raw = float(input[i]);
      float x = clamp(x_raw, 1e-7f, 1.f - 1e-7f);
      float dx = (x_raw - float(target[i])) / (x * (1.f - x));
      if (p.has_weight)
        dx *= float(weight[i]);
      float g = (p.reduction == 0) ? float(grad_out[i]) * dx : g_scalar * dx;
      grad_in[i] = T(g);
    }
  }
}

#define INSTANTIATE_BCE(T)                          \
  template [[host_name("bce_loss_fwd_none_" #T)]]   \
  kernel void bce_loss_fwd_none<T>(                 \
      device const T*,                              \
      device const T*,                              \
      device const T*,                              \
      device T*,                                    \
      constant BCEParams&,                          \
      uint);                                        \
  template [[host_name("bce_loss_fwd_reduce_" #T)]] \
  kernel void bce_loss_fwd_reduce<T>(               \
      device const T*,                              \
      device const T*,                              \
      device const T*,                              \
      device float*,                                \
      constant BCEParams&,                          \
      uint,                                         \
      uint,                                         \
      uint,                                         \
      uint,                                         \
      uint);                                        \
  template [[host_name("bce_loss_bwd_" #T)]]        \
  kernel void bce_loss_bwd<T>(                      \
      device const T*,                              \
      device const T*,                              \
      device const T*,                              \
      device const T*,                              \
      device T*,                                    \
      constant BCEParams&,                          \
      uint);

INSTANTIATE_BCE(float)
INSTANTIATE_BCE(half)
INSTANTIATE_BCE(bfloat)
