//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/binary_cross_entropy_backward_native.h>
#include <ATen/ops/binary_cross_entropy_native.h>
#include <ATen/ops/huber_loss_backward_native.h>
#include <ATen/ops/huber_loss_native.h>
#include <ATen/ops/mse_loss_backward_native.h>
#include <ATen/ops/mse_loss_native.h>
#include <ATen/ops/nll_loss2d_backward_native.h>
#include <ATen/ops/nll_loss2d_forward_native.h>
#include <ATen/ops/nll_loss_backward_native.h>
#include <ATen/ops/nll_loss_forward_native.h>
#include <ATen/ops/smooth_l1_loss_backward_native.h>
#include <ATen/ops/smooth_l1_loss_native.h>
#endif

namespace at::native {
namespace mps {

// ── Metal kernel library (LossOps.metal) ────────────────────────────────────
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LossOps_metallib.h>
#endif

struct BCEParams {
  uint32_t N;
  float scale;
  uint32_t reduction;
  uint32_t has_weight;
};

static constexpr uint32_t kLossKernelTgsz = 256;
// Cap reduce threadgroup count: avoids O(N/256) barrier calls at large N
static constexpr uint32_t kMaxReduceTGs = 256u;

static void encode_reduce_partials(id<MTLComputeCommandEncoder> enc,
                                   const Tensor& partial,
                                   const Tensor& loss_out,
                                   uint32_t n_tg) {
  const std::string dt_out = scalarToMetalTypeString(loss_out);
  auto pso = lib.getPipelineStateForFunc("loss_reduce_partials_" + dt_out);
  [enc setComputePipelineState:pso];
  mtl_setArgs(enc, partial, loss_out, n_tg);
  [enc dispatchThreads:MTLSizeMake(kLossKernelTgsz, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
}

// Binary cross-entropy via Metal kernels (forward/backward), replacing the
// shape-dependent MPSGraph cache. "loss" is the grad_input on the backward
// pass (when grad_output_opt is set).
static Tensor& bce_loss_metal(const Tensor& input,
                              const Tensor& target,
                              const std::optional<Tensor>& weight_opt,
                              int64_t reduction,
                              Tensor& loss,
                              const std::optional<Tensor>& grad_output_opt,
                              const std::string& op_name) {
  TORCH_CHECK(target.is_same_size(input), op_name + ": target and input tensors must have identical shapes");
  const bool is_bwd = grad_output_opt.has_value();
  c10::MaybeOwned<Tensor> wmo = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *wmo;
  loss.resize_((reduction == Reduction::None || is_bwd) ? target.sizes() : IntArrayRef({}));
  TORCH_CHECK(loss.is_mps());
  const uint32_t N = static_cast<uint32_t>(input.numel());
  if (N == 0) {
    // Empty input: sum of no terms is 0, mean is NaN; None/backward keep the empty tensor.
    if (!is_bwd && reduction != Reduction::None)
      loss.fill_(reduction == Reduction::Mean ? std::numeric_limits<float>::quiet_NaN() : 0.0f);
    return loss;
  }
  const float scale = (reduction == Reduction::Mean) ? 1.f / static_cast<float>(N) : 1.f;
  BCEParams p{N, scale, static_cast<uint32_t>(reduction), weight.defined() ? 1u : 0u};
  const std::string dt = scalarToMetalTypeString(input);
  MPSStream* stream = getCurrentMPSStream();
  // Expand weight to input shape: Metal kernel reads weight[i] linearly for i in [0, N)
  const Tensor wt = weight.defined()
      ? (weight.sizes() == input.sizes() ? (weight.is_contiguous() ? weight : weight.contiguous())
                                         : weight.expand_as(input).contiguous())
      : input;
  if (is_bwd) {
    // grad_out may have stride-0 layout (e.g. from sum().backward() → expand)
    const Tensor grad_out_c =
        grad_output_opt.value().is_contiguous() ? grad_output_opt.value() : grad_output_opt.value().contiguous();
    const Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor target_c = target.is_contiguous() ? target : target.contiguous();
    // grad_input ("loss") may be non-contiguous (e.g. an out= into a transposed
    // view): the kernel writes linearly, so reduce into a contiguous buffer and
    // copy back, mirroring the forward path.
    const bool need_loss_copy = !loss.is_contiguous();
    Tensor loss_buf = need_loss_copy ? at::empty_like(loss, at::MemoryFormat::Contiguous) : loss;
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("bce_loss_bwd_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, grad_out_c, input_c, target_c, wt, loss_buf, p);
        [enc dispatchThreads:MTLSizeMake((N + 15u) / 16u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
      }
    });
    stream->synchronize(SyncType::COMMIT);
    if (need_loss_copy)
      loss.copy_(loss_buf);
    return loss;
  }
  // Forward path: ensure input/target are contiguous before passing to kernel.
  const Tensor input_c = input.is_contiguous() ? input : input.contiguous();
  const Tensor target_c = target.is_contiguous() ? target : target.contiguous();
  const bool need_loss_copy = !loss.is_contiguous();
  Tensor loss_buf = need_loss_copy ? at::empty_like(loss, at::MemoryFormat::Contiguous) : loss;
  if (reduction == Reduction::None) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("bce_loss_fwd_none_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input_c, target_c, wt, loss_buf, p);
        [enc dispatchThreads:MTLSizeMake((N + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
      }
    });
    stream->synchronize(SyncType::COMMIT);
  } else {
    const uint32_t n_tg = std::min((N + kLossKernelTgsz - 1) / kLossKernelTgsz, kMaxReduceTGs);
    Tensor partial = at::empty({static_cast<int64_t>(n_tg)}, input.options().dtype(at::kFloat));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("bce_loss_fwd_reduce_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input_c, target_c, wt, partial, p);
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
        encode_reduce_partials(enc, partial, loss, n_tg);
      }
    });
    stream->synchronize(SyncType::COMMIT);
  }
  if (need_loss_copy)
    loss.copy_(loss_buf);
  return loss;
}

static std::string reductionToString(int64_t reduction) {
  switch (reduction) {
    case Reduction::Mean:
      return "Mean";
    case Reduction::Sum:
      return "Sum";
    default:
      return "None";
  }
}

static MPSGraphTensor* reduceTensor(MPSGraphTensor* tensor,
                                    int64_t reduction,
                                    MPSGraph* mpsGraph,
                                    NSUInteger axesCount) {
  NSMutableArray<NSNumber*>* axes = [NSMutableArray<NSNumber*> arrayWithCapacity:axesCount];
  for (NSUInteger i = 0; i < axesCount; i++)
    axes[i] = @(i);

  switch (reduction) {
    case Reduction::Mean:
      return [mpsGraph meanOfTensor:tensor axes:axes name:@"reductionMeanTensor"];
    case Reduction::Sum:
      return [mpsGraph reductionSumWithTensor:tensor axes:axes name:@"reductionSumTensor"];
    default:
      assert(reduction == Reduction::None);
      return tensor;
  }
}

static Tensor& mse_loss_backward_out_impl(const Tensor& grad_output,
                                          const Tensor& input,
                                          const Tensor& target,
                                          int64_t reduction,
                                          Tensor& grad_input,
                                          const std::string& op_name) {
  TORCH_CHECK(target.is_same_size(input), op_name + ": target and input tensors must have identical shapes")
  auto norm = reduction == Reduction::Mean ? 2. / static_cast<double>(input.numel()) : 2.;

  if ((input.numel() == 0) || (target.numel() == 0) || (grad_output.numel() == 0)) {
    reduction == Reduction::Mean ? grad_input.fill_(std::numeric_limits<float>::quiet_NaN()) : grad_input.zero_();
    return grad_input;
  }
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *targetTensor = nil;
    MPSGraphTensor *gradInputTensor = nil, *gradOutputTensor = nil;
  };

  @autoreleasepool {
    std::string key = op_name + reductionToString(reduction) + ":" + std::to_string(grad_input.sizes()[1]) +
        getTensorsStringKey({input, target, grad_output});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      newCachedGraph->targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, target);
      newCachedGraph->gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* normTensor = [mpsGraph constantWithScalar:norm dataType:[newCachedGraph->inputTensor dataType]];
      MPSGraphTensor* diffTensor = [mpsGraph subtractionWithPrimaryTensor:newCachedGraph->inputTensor
                                                          secondaryTensor:newCachedGraph->targetTensor
                                                                     name:nil];
      MPSGraphTensor* diffGradientTensor = [mpsGraph multiplicationWithPrimaryTensor:diffTensor
                                                                     secondaryTensor:newCachedGraph->gradOutputTensor
                                                                                name:nil];
      newCachedGraph->gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:diffGradientTensor
                                                                  secondaryTensor:normTensor
                                                                             name:nil];
    });
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor, input);
    Placeholder targetPlaceholder = Placeholder(cachedGraph->targetTensor, target);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor, grad_input);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor, grad_output);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder, targetPlaceholder, gradOutputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, gradInputPlaceholder);
  }

  return grad_input;
}

static inline MPSGraphTensor* divisionNoNaN(MPSGraph* mpsGraph, MPSGraphTensor* divident, MPSGraphTensor* divisor) {
  auto* div = [mpsGraph divisionWithPrimaryTensor:divident
                                  secondaryTensor:castMPSTensor(mpsGraph, divisor, divident.dataType)
                                             name:@"divisionTensor"];
  // Replace NaNs with 0 for divident elements equal to 0
  return [mpsGraph selectWithPredicateTensor:castMPSTensor(mpsGraph, divisor, MPSDataTypeBool)
                         truePredicateTensor:div
                        falsePredicateTensor:[mpsGraph constantWithScalar:0.0 dataType:div.dataType]
                                        name:nil];
}

// NLLLoss
static void nllnd_loss_backward_impl(Tensor& grad_input_arg,
                                     const Tensor& grad_output_arg,
                                     const Tensor& input_arg,
                                     const Tensor& target_arg,
                                     const Tensor& weight_arg,
                                     int64_t reduction,
                                     int64_t ignore_index,
                                     const Tensor& total_weight,
                                     bool is2D) {
  if (grad_input_arg.numel() == 0) {
    return;
  }
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* targetTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* totalWeightTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
    MPSGraphTensor* gradOutputTensor_ = nil;
  };
  bool isWeightsArrayValid = weight_arg.defined() && weight_arg.numel() > 0;
  bool isTargetCasted = target_arg.scalar_type() != ScalarType::Long;
  int64_t channel_dim = grad_input_arg.dim() < 2 ? 0 : 1;
  auto input = input_arg.dim() == 1 ? input_arg.view({1, input_arg.size(0)}) : input_arg;
  auto target = target_arg.dim() == 0 ? target_arg.view({1}) : target_arg;
  auto grad_input = grad_input_arg.dim() == 1 ? grad_input_arg.view({1, grad_input_arg.size(0)}) : grad_input_arg;
  auto numClasses = grad_input.sizes()[1];
  auto weight = weight_arg;
  auto grad_output = grad_output_arg;

  if (isWeightsArrayValid) {
    std::vector<int64_t> weightShape(input.dim(), 1);
    weightShape[1] = input.size(1);
    weight = weight_arg.view(weightShape);
  }
  if (grad_output_arg.dim() < grad_input.dim() && grad_output_arg.dim() > 0) {
    grad_output = grad_output_arg.unsqueeze(channel_dim);
  }
  @autoreleasepool {
    std::string key = "nllnd_loss_backward" + getTensorsStringKey({input, grad_output, target, weight, total_weight}) +
        std::to_string(numClasses) + ":" + std::to_string(ignore_index) + ":" + std::to_string(isWeightsArrayValid) +
        ":" + std::to_string(isTargetCasted) + ":" + reductionToString(reduction);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, target);
      MPSGraphTensor* castedTargetTensor =
          isTargetCasted ? castMPSTensor(mpsGraph, targetTensor, MPSDataTypeInt64) : targetTensor;
      MPSGraphTensor* weightTensor = nil;
      if (isWeightsArrayValid) {
        weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight);
      }
      MPSGraphTensor* totalWeightTensor = mpsGraphRankedPlaceHolder(mpsGraph, total_weight);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* updatedTargetTensor = castedTargetTensor;

      // Replace ignored_index with length depth + 1 so that oneHotAPI ignores it
      MPSGraphTensor* ignoreIndexTensor = [mpsGraph constantWithScalar:ignore_index dataType:MPSDataTypeInt64];
      MPSGraphTensor* numClassesTensor = [mpsGraph constantWithScalar:(numClasses + 1) dataType:MPSDataTypeInt64];
      MPSGraphTensor* isEqualTensor = [mpsGraph equalWithPrimaryTensor:castedTargetTensor
                                                       secondaryTensor:ignoreIndexTensor
                                                                  name:@"isEqualTensor"];
      updatedTargetTensor = [mpsGraph selectWithPredicateTensor:isEqualTensor
                                            truePredicateTensor:numClassesTensor
                                           falsePredicateTensor:castedTargetTensor
                                                           name:@"predicateTensor"];

      // oneHotWithIndicesTensor only works for Float32 dtype
      // cast it explicitly later if needed
      auto* oneHotTensor = [mpsGraph oneHotWithIndicesTensor:updatedTargetTensor
                                                       depth:numClasses
                                                        axis:1
                                                    dataType:MPSDataTypeFloat32
                                                     onValue:-1.0f
                                                    offValue:0.0f
                                                        name:nil];
      oneHotTensor = castMPSTensor(mpsGraph, oneHotTensor, [inputTensor dataType]);
      if (isWeightsArrayValid) {
        oneHotTensor = [mpsGraph multiplicationWithPrimaryTensor:oneHotTensor
                                                 secondaryTensor:weightTensor
                                                            name:@"scaleByWeightTensor"];
      }
      if (reduction == Reduction::Mean) {
        oneHotTensor = divisionNoNaN(mpsGraph, oneHotTensor, totalWeightTensor);
      }
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:oneHotTensor
                                                                  secondaryTensor:gradOutputTensor
                                                                             name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->targetTensor_ = targetTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->totalWeightTensor_ = totalWeightTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    auto targetPlaceholder = Placeholder(cachedGraph->targetTensor_, target);
    Placeholder weightPlaceholder = Placeholder();
    if (isWeightsArrayValid) {
      weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight);
    }
    auto totalWeightPlaceholder = Placeholder(cachedGraph->totalWeightTensor_, total_weight);
    auto gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[targetPlaceholder.getMPSGraphTensor()] = targetPlaceholder.getMPSGraphTensorData();
    feeds[totalWeightPlaceholder.getMPSGraphTensor()] = totalWeightPlaceholder.getMPSGraphTensorData();
    feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();

    if (isWeightsArrayValid) {
      feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
    }
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

static void nllnd_loss_forward_impl(Tensor& output,
                                    Tensor& total_weight,
                                    const Tensor& input_arg,
                                    const Tensor& target_arg,
                                    const Tensor& weight,
                                    int64_t reduction,
                                    int64_t ignore_index,
                                    bool is2D) {
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(output.scalar_type()),
                              "nlld_loss for complex is not supported for MPS");
  if (weight.defined()) {
    TORCH_CHECK(input_arg.scalar_type() == weight.scalar_type(),
                "expected scalar type ",
                input_arg.scalar_type(),
                " but found ",
                weight.scalar_type());
  }
  std::vector<long long> reshapedTarget(target_arg.sizes().begin(), target_arg.sizes().end());
  reshapedTarget.push_back(1);

  Tensor batchSizeTensor = at::empty_like(input_arg).resize_(IntArrayRef(1));
  float batchVal = 1.0f;
  for (size_t i = 0; i < reshapedTarget.size(); ++i)
    batchVal *= reshapedTarget[i];
  batchSizeTensor[0] = batchVal;

  if (reduction == Reduction::None)
    output.resize_(target_arg.sizes());
  if (reduction == Reduction::Sum)
    output.resize_({});
  if (reduction == Reduction::Mean)
    output.resize_({});

  TORCH_CHECK(output.is_mps());

  // Empty output
  if (output.numel() == 0)
    return;

  // https://github.com/pytorch/pytorch/blob/042f2f7746a064f1527d95d1f1d712b4f0b34186/aten/src/ATen/native/cuda/Loss.cu#L335-L346
  if (target_arg.numel() == 0) {
    // Here target (and input) have zero elements
    // Mean reduction on empty tensors produces NaN. See the discussion in
    // https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if (reduction == Reduction::Mean) {
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      output.zero_();
    }
    total_weight.zero_();
    return;
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* targetTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* batchSizeTensor_ = nil;
    MPSGraphTensor* totalWeightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  auto input = input_arg.dim() == 1 ? input_arg.view({1, input_arg.size(0)}) : input_arg;
  auto target = target_arg.dim() == 0 ? target_arg.view({1}) : target_arg;

  @autoreleasepool {
    bool isWeightsArrayValid = (weight.numel() > 0);
    bool isTargetCasted = target.scalar_type() != ScalarType::Long;

    MPSShape* input_shape = getMPSShape(input);
    MPSShape* target_shape = getMPSShape(target);
    MPSShape* weight_shape = getMPSShape(weight);

    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    // TODO: Make the key
    std::string key = "nllnd_loss_forward_impl:" + std::to_string(ignore_index) + ":" +
        std::to_string(isWeightsArrayValid) + ":" + reductionToString(reduction) + ":" + [ns_shape_key UTF8String] +
        ":" + getMPSTypeString(input) + ":" + getMPSTypeString(target) + ":" + std::to_string(isTargetCasted) + ":" +
        getMPSTypeString(weight);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), input_shape);
      MPSGraphTensor* targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(target), target_shape);
      MPSGraphTensor* castedTargetTensor =
          isTargetCasted ? castMPSTensor(mpsGraph, targetTensor, MPSDataTypeInt64) : targetTensor;
      MPSGraphTensor* weightTensor = nil;
      if (isWeightsArrayValid)
        weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(weight), weight_shape);
      MPSGraphTensor* mps_batchSizeTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(batchSizeTensor));

      MPSGraphTensor* mpsGraphBatchSizeTensor = mps_batchSizeTensor;

      // The transposes are needed to get the class dimension (dim 1) to the inner most dim for gather op.
      // The transpose become nop in the 2D case.
      MPSGraphTensor* mpsTransposeTensor = inputTensor;
      int classDim = 1;
      int lastDim = input.sizes().size() - 1;
      mpsTransposeTensor = [mpsGraph transposeTensor:inputTensor dimension:classDim withDimension:lastDim name:nil];
      for (int i = 0; i < lastDim - 2; ++i) {
        mpsTransposeTensor = [mpsGraph transposeTensor:mpsTransposeTensor
                                             dimension:classDim + i
                                         withDimension:classDim + i + 1
                                                  name:nil];
      }

      MPSGraphTensor* mpsGatherTensor = [mpsGraph gatherWithUpdatesTensor:mpsTransposeTensor
                                                            indicesTensor:castedTargetTensor
                                                                     axis:lastDim
                                                          batchDimensions:lastDim
                                                                     name:@"gatherTensor"];

      MPSGraphTensor* mpsGraphZeroTensor = [mpsGraph constantWithScalar:0.0 dataType:mpsGatherTensor.dataType];
      MPSGraphTensor* mpsGraphOneTensor = [mpsGraph constantWithScalar:1.0 dataType:mpsGatherTensor.dataType];
      MPSGraphTensor* mpsGraphIndexTensor = [mpsGraph constantWithScalar:ignore_index dataType:MPSDataTypeInt64];
      MPSGraphTensor* mpsGraphIsEqualTensor = [mpsGraph equalWithPrimaryTensor:castedTargetTensor
                                                               secondaryTensor:mpsGraphIndexTensor
                                                                          name:@"isEqualTensor"];
      // Zero out loss
      mpsGatherTensor = [mpsGraph selectWithPredicateTensor:mpsGraphIsEqualTensor
                                        truePredicateTensor:mpsGraphZeroTensor
                                       falsePredicateTensor:mpsGatherTensor
                                                       name:@"predicateTensor"];

      if (isWeightsArrayValid) {
        MPSGraphTensor* weightGatherTensor = [mpsGraph gatherWithUpdatesTensor:weightTensor
                                                                 indicesTensor:castedTargetTensor
                                                                          axis:0
                                                               batchDimensions:0
                                                                          name:@"weightGatherTensor"];
        mpsGatherTensor = [mpsGraph multiplicationWithPrimaryTensor:weightGatherTensor
                                                    secondaryTensor:mpsGatherTensor
                                                               name:@"scaledLossTensor"];
        mpsGraphOneTensor = weightGatherTensor;
      }

      // Compute new batch size
      MPSGraphTensor* mpsSelectOneTensor = [mpsGraph selectWithPredicateTensor:mpsGraphIsEqualTensor
                                                           truePredicateTensor:mpsGraphZeroTensor
                                                          falsePredicateTensor:mpsGraphOneTensor
                                                                          name:@"predicateOneTensor"];

      MPSGraphTensor* mpsGraphNegTensor = [mpsGraph negativeWithTensor:mpsGatherTensor name:@"negativeTensor"];

      MPSGraphTensor* mpsGraphReducedTensor = mpsGraphNegTensor;

      if (!(reduction == Reduction::None)) {
        mpsGraphReducedTensor = [mpsGraph reductionSumWithTensor:mpsGraphNegTensor axes:nil name:@"reductionSumTensor"];
        if (reduction == Reduction::Mean) {
          mpsGraphBatchSizeTensor = [mpsGraph reductionSumWithTensor:mpsSelectOneTensor
                                                                axes:nil
                                                                name:@"batchSizeReductionTensor"];
          mpsGraphReducedTensor = [mpsGraph divisionWithPrimaryTensor:mpsGraphReducedTensor
                                                      secondaryTensor:mpsGraphBatchSizeTensor
                                                                 name:@"divisionTensor"];
        }
      }

      mpsGraphReducedTensor = [mpsGraph reshapeTensor:mpsGraphReducedTensor withShape:getMPSShape(output) name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->targetTensor_ = targetTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->batchSizeTensor_ = mps_batchSizeTensor;
      newCachedGraph->totalWeightTensor_ = mpsGraphBatchSizeTensor;
      newCachedGraph->outputTensor_ = mpsGraphReducedTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, input, nil, true, MPSDataTypeInvalid, false);
    Placeholder targetPlaceholder =
        Placeholder(cachedGraph->targetTensor_, target, nil, true, MPSDataTypeInvalid, false);
    Placeholder weightPlaceholder = Placeholder();
    if (isWeightsArrayValid)
      weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight, nil, true, MPSDataTypeInvalid, false);
    Placeholder batchSizePlaceholder =
        Placeholder(cachedGraph->batchSizeTensor_, batchSizeTensor, nil, true, MPSDataTypeInvalid, false);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, output, nil, true, MPSDataTypeInvalid, false);
    Placeholder totalWeightsPlaceholder =
        Placeholder(cachedGraph->totalWeightTensor_, total_weight, nil, true, MPSDataTypeInvalid, false);

    // Create dictionary of inputs and outputs
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        [[[NSMutableDictionary alloc] initWithCapacity:4] autorelease];
    feeds[selfPlaceholder.getMPSGraphTensor()] = selfPlaceholder.getMPSGraphTensorData();
    feeds[targetPlaceholder.getMPSGraphTensor()] = targetPlaceholder.getMPSGraphTensorData();
    feeds[batchSizePlaceholder.getMPSGraphTensor()] = batchSizePlaceholder.getMPSGraphTensorData();

    if (isWeightsArrayValid)
      feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();

    auto results = dictionaryFromPlaceholders(outputPlaceholder, totalWeightsPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return;
}

static void smooth_l1_loss_impl(const Tensor& input,
                                const Tensor& target,
                                const int64_t reduction,
                                double beta,
                                const Tensor& output,
                                MPSShape* mpsInputShape,
                                MPSShape* mpsOutputShape) {
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* targetTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* input_shape = getMPSShape(input);
    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    std::string key = "smooth_l1_loss_impl:" + reductionToString(reduction) + ":" + [ns_shape_key UTF8String] + ":" +
        std::to_string(beta) + ":" + getMPSTypeString(input) + ":" + getMPSTypeString(target);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      // smooth_l1_loss_mps:
      // ln = 0.5 * ( xn - yn ) ^ 2 / beta,       if |xn - yn| < beta
      //    = | xn - yn | - 0.5 * beta,           otherwise

      MPSGraphTensor* inputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(input));
      MPSGraphTensor* targetTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(target));

      // Setup tensors
      MPSGraphTensor* mpsGraphHalfTensor = [mpsGraph constantWithScalar:0.5 dataType:inputTensor.dataType];
      MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta dataType:inputTensor.dataType];
      // 0.5 * beta
      MPSGraphTensor* halfTensorMulBetaTensor = [mpsGraph constantWithScalar:beta * 0.5 dataType:inputTensor.dataType];
      // Calculating first part of the equation:
      // ln = 0.5(xn - yn)^2/beta, if |xn - yn| < beta

      // xn - yn
      MPSGraphTensor* diffTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                          secondaryTensor:targetTensor
                                                                     name:nil];

      // | xn - yn |
      MPSGraphTensor* diffAbsTensor = [mpsGraph absoluteWithTensor:diffTensor name:nil];

      // | xn - yn | < beta
      MPSGraphTensor* diffAbsLessThanBetaTensor = [mpsGraph lessThanWithPrimaryTensor:diffAbsTensor
                                                                      secondaryTensor:betaTensor
                                                                                 name:nil];

      // ( xn - yn ) ^ 2
      MPSGraphTensor* diffSquare = [mpsGraph squareWithTensor:diffTensor name:nil];

      // 0.5 * ( xn - yn ) ^ 2
      MPSGraphTensor* diffSquareMulHalfTensor = [mpsGraph multiplicationWithPrimaryTensor:diffSquare
                                                                          secondaryTensor:mpsGraphHalfTensor
                                                                                     name:nil];

      // 0.5 * ( xn - yn ) ^ 2 / beta
      MPSGraphTensor* loss1Temp = [mpsGraph divisionWithPrimaryTensor:diffSquareMulHalfTensor
                                                      secondaryTensor:betaTensor
                                                                 name:nil];

      // Calculating second part of the equation:
      // | xn - yn | - 0.5 * beta, if | xn - yn | >= beta

      // | xn - yn | - 0.5 * beta
      MPSGraphTensor* loss2Temp = [mpsGraph subtractionWithPrimaryTensor:diffAbsTensor
                                                         secondaryTensor:halfTensorMulBetaTensor
                                                                    name:nil];

      MPSGraphTensor* lossTensor = [mpsGraph selectWithPredicateTensor:diffAbsLessThanBetaTensor
                                                   truePredicateTensor:loss1Temp
                                                  falsePredicateTensor:loss2Temp
                                                                  name:@"lossTensor"];

      MPSGraphTensor* outputTensor = reduceTensor(lossTensor, reduction, mpsGraph, 1);

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->targetTensor_ = targetTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input, mpsInputShape);
    Placeholder targetPlaceholder = Placeholder(cachedGraph->targetTensor_, target, mpsInputShape);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output, mpsOutputShape);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder, targetPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void smooth_l1_loss_template(const Tensor& input,
                                    const Tensor& target,
                                    const int64_t reduction,
                                    double beta,
                                    const Tensor& output) {
  TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.");
  TORCH_CHECK(input.is_mps());
  TORCH_CHECK(target.is_mps());
  TORCH_CHECK_NOT_IMPLEMENTED(input.scalar_type() != kLong, "MPS doesn't know how to do square_i64");
  if ((input.numel() == 0) || (target.numel() == 0)) {
    reduction == Reduction::Mean ? output.fill_(std::numeric_limits<float>::quiet_NaN()) : output.zero_();
    return;
  }
  MPSShape* mpsInputShape = nil;
  MPSShape* mpsOutputShape = nil;

  // Determine the shape of the output
  // If the reduction is 'mean' or 'sum', the output shape is a scalar,
  // otherwise, the output shape is the same shape as input
  if (reduction == Reduction::Mean || reduction == Reduction::Sum) {
    // Output: scalar, if reduction is 'mean' or 'sum'
    IntArrayRef input_shape = input.sizes();
    int64_t num_input_dims = input_shape.size();
    NSMutableArray<NSNumber*>* apparent_input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    int64_t num_in_elements = 1;
    for (int i = 0; i < num_input_dims; i++) {
      num_in_elements *= input_shape[i];
    }
    apparent_input_shape[0] = [NSNumber numberWithInt:num_in_elements];

    // Output is a single value in case reduction is set to mean or sum
    NSMutableArray<NSNumber*>* apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    apparent_out_shape[0] = @1;
    mpsInputShape = apparent_input_shape;
    mpsOutputShape = apparent_out_shape;
  } else {
    // Output: If reduction is 'none', then (N, *); same shape as the input
    assert(reduction == Reduction::None);
    mpsInputShape = getMPSShape(input);
    mpsOutputShape = mpsInputShape;
    // resize_tensor(&output);
  }
  TORCH_CHECK(output.is_mps());

  smooth_l1_loss_impl(input, target, reduction, beta, output, mpsInputShape, mpsOutputShape);
}

static void smooth_l1_loss_backward_impl(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         int64_t reduction,
                                         double beta,
                                         Tensor& grad_input) {
  if (grad_input.numel() == 0) {
    return;
  }
  TORCH_CHECK(beta >= 0, "smooth_l1_loss_backward does not support negative values for beta.");

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* targetTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
    MPSGraphTensor* gradOutputTensor_ = nil;
  };

  @autoreleasepool {
    std::string key = "smooth_l1_loss_backward" + getTensorsStringKey({input, grad_output, grad_input, target}) + ":" +
        reductionToString(reduction) + ":" + std::to_string(beta);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, target);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta dataType:[inputTensor dataType]];
      // xn - yn
      MPSGraphTensor* diffTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                          secondaryTensor:targetTensor
                                                                     name:nil];
      // | xn - yn |
      MPSGraphTensor* diffAbsTensor = [mpsGraph absoluteWithTensor:diffTensor name:nil];
      // | xn - yn | < beta
      MPSGraphTensor* diffAbsLessThanBetaTensor = [mpsGraph lessThanWithPrimaryTensor:diffAbsTensor
                                                                      secondaryTensor:betaTensor
                                                                                 name:nil];
      // ( xn - yn ) / beta
      MPSGraphTensor* truePredicateTensor = [mpsGraph divisionWithPrimaryTensor:diffTensor
                                                                secondaryTensor:betaTensor
                                                                           name:nil];
      // ( x - y ) / | x - y |
      MPSGraphTensor* falsePredicateTensor = [mpsGraph divisionWithPrimaryTensor:diffTensor
                                                                 secondaryTensor:diffAbsTensor
                                                                            name:nil];

      MPSGraphTensor* lossTensor = [mpsGraph selectWithPredicateTensor:diffAbsLessThanBetaTensor
                                                   truePredicateTensor:truePredicateTensor
                                                  falsePredicateTensor:falsePredicateTensor
                                                                  name:@"lossTensor"];
      MPSGraphTensor* outputTensor = lossTensor;
      if (reduction == Reduction::Mean) {
        MPSGraphTensor* numelTensor = [mpsGraph constantWithScalar:(double)input.numel()
                                                          dataType:[lossTensor dataType]];
        outputTensor = [mpsGraph divisionWithPrimaryTensor:lossTensor secondaryTensor:numelTensor name:nil];
      }
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:outputTensor
                                                                  secondaryTensor:gradOutputTensor
                                                                             name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->targetTensor_ = targetTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
    });
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder targetPlaceholder = Placeholder(cachedGraph->targetTensor_, target);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder, targetPlaceholder, gradOutputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

} // namespace mps

// APIs exposed to at::native scope

// HuberLoss

Tensor& huber_loss_out_mps(const Tensor& input, const Tensor& target, int64_t reduction, double delta, Tensor& output) {
  std::string op_name = __func__;
  using namespace mps;
  TORCH_CHECK_NOT_IMPLEMENTED(input.scalar_type() != kLong, "MPS doesn't know how to do square_i64");
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input.scalar_type()),
                              "huber_loss for complex is not supported for MPS");
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
  TORCH_CHECK(target.is_same_size(input), op_name + ": target and input tensors must have identical shapes")
  TORCH_CHECK(output.is_mps());

  if (reduction == Reduction::None)
    output.resize_(target.sizes());
  if (reduction == Reduction::Sum)
    output.resize_({});
  if (reduction == Reduction::Mean)
    output.resize_({});

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* targetTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    std::string key = op_name + ":" + reductionToString(reduction) + ":" + std::to_string(delta) + ":" +
        getTensorsStringKey({input, target});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, target);

      MPSDataType input_type = getMPSScalarType(input.scalar_type());
      MPSGraphTensor* deltaTensor = [mpsGraph constantWithScalar:delta shape:@[ @1 ] dataType:input_type];
      MPSGraphTensor* halfTensor = [mpsGraph constantWithScalar:.5f shape:@[ @1 ] dataType:input_type];

      MPSGraphTensor* diffTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                          secondaryTensor:targetTensor
                                                                     name:nil];
      MPSGraphTensor* absDiffTensor = [mpsGraph absoluteWithTensor:diffTensor name:nil];
      MPSGraphTensor* firstCondTensor = [mpsGraph multiplicationWithPrimaryTensor:absDiffTensor
                                                                  secondaryTensor:absDiffTensor
                                                                             name:nil];
      firstCondTensor = [mpsGraph multiplicationWithPrimaryTensor:firstCondTensor secondaryTensor:halfTensor name:nil];
      MPSGraphTensor* secondCondTensor = [mpsGraph multiplicationWithPrimaryTensor:deltaTensor
                                                                   secondaryTensor:halfTensor
                                                                              name:nil];
      secondCondTensor = [mpsGraph subtractionWithPrimaryTensor:absDiffTensor
                                                secondaryTensor:secondCondTensor
                                                           name:nil];
      secondCondTensor = [mpsGraph multiplicationWithPrimaryTensor:deltaTensor
                                                   secondaryTensor:secondCondTensor
                                                              name:nil];
      MPSGraphTensor* outputTensor =
          [mpsGraph selectWithPredicateTensor:[mpsGraph lessThanOrEqualToWithPrimaryTensor:absDiffTensor
                                                                           secondaryTensor:deltaTensor
                                                                                      name:nil]
                          truePredicateTensor:firstCondTensor
                         falsePredicateTensor:secondCondTensor
                                         name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->targetTensor_ = targetTensor;
      newCachedGraph->outputTensor_ = reduceTensor(outputTensor, reduction, mpsGraph, input.sizes().size());
    });
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder targetPlaceholder = Placeholder(cachedGraph->targetTensor_, target);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder, targetPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return output;
}

Tensor huber_loss_mps(const Tensor& input, const Tensor& target, int64_t reduction, double delta) {
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.");
  Tensor output = at::empty(input.sizes(), input.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  return huber_loss_out_mps(input, target, reduction, delta, output);
}

Tensor& huber_loss_backward_out_mps(const Tensor& grad_output,
                                    const Tensor& input,
                                    const Tensor& target,
                                    int64_t reduction,
                                    double delta,
                                    Tensor& grad_input) {
  using namespace mps;
  auto is_mean_reduction = reduction == Reduction::Mean;
  auto input_numel = input.numel();

  auto new_grad_output = grad_output.contiguous();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* targetTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* input_shape = getMPSShape(input);
    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    std::string key = "huber_loss_backward_out_mps:" + reductionToString(reduction) + ":" + std::to_string(delta) +
        ":" + [ns_shape_key UTF8String] + ":" + getMPSTypeString(input) + ":" + getMPSTypeString(target);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(new_grad_output), getMPSShape(new_grad_output));
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), input_shape);
      MPSGraphTensor* targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(target), getMPSShape(target));
      MPSGraphTensor* isMeanReductionTensor =
          [mpsGraph constantWithScalar:is_mean_reduction
                              dataType:MPSDataTypeInt64]; // constant does not support MPSDataTypeBool
      MPSGraphTensor* inputNumelTensor = [mpsGraph constantWithScalar:input_numel
                                                             dataType:getMPSDataType(new_grad_output)];

      MPSGraphTensor* normGradOutputTensor =
          [mpsGraph selectWithPredicateTensor:isMeanReductionTensor
                          truePredicateTensor:[mpsGraph divisionWithPrimaryTensor:gradOutputTensor
                                                                  secondaryTensor:inputNumelTensor
                                                                             name:nil]
                         falsePredicateTensor:gradOutputTensor
                                         name:nil];
      MPSGraphTensor* deltaTensor = [mpsGraph constantWithScalar:delta
                                                           shape:getMPSShape(target)
                                                        dataType:getMPSDataType(target)];
      MPSGraphTensor* diffTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                          secondaryTensor:targetTensor
                                                                     name:nil];
      MPSGraphTensor* normGradOutputDeltaTensor = [mpsGraph multiplicationWithPrimaryTensor:normGradOutputTensor
                                                                            secondaryTensor:deltaTensor
                                                                                       name:nil];
      // first condition: (input - target) <= -delta
      // formula: -norm * grad_output * delta
      MPSGraphTensor* firstCondTensor = [mpsGraph negativeWithTensor:normGradOutputDeltaTensor name:nil];
      // second condition: (input - target) >= delta
      // formula: norm * grad_output * delta
      MPSGraphTensor* secondCondTensor = normGradOutputDeltaTensor;

      // third condition: (input - target) within -delta to delta
      // formula: norm * (input - target) * grad_output
      MPSGraphTensor* thirdCondTensor = [mpsGraph multiplicationWithPrimaryTensor:normGradOutputTensor
                                                                  secondaryTensor:diffTensor
                                                                             name:nil];

      MPSGraphTensor* secondThirdTensor =
          [mpsGraph selectWithPredicateTensor:[mpsGraph greaterThanOrEqualToWithPrimaryTensor:diffTensor
                                                                              secondaryTensor:deltaTensor
                                                                                         name:nil]
                          truePredicateTensor:secondCondTensor
                         falsePredicateTensor:thirdCondTensor
                                         name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph
          selectWithPredicateTensor:[mpsGraph
                                        lessThanOrEqualToWithPrimaryTensor:diffTensor
                                                           secondaryTensor:[mpsGraph negativeWithTensor:deltaTensor
                                                                                                   name:nil]
                                                                      name:nil]
                truePredicateTensor:firstCondTensor
               falsePredicateTensor:secondThirdTensor
                               name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->targetTensor_ = targetTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, new_grad_output);
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder targetPlaceholder = Placeholder(cachedGraph->targetTensor_, target);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, inputPlaceholder, targetPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return grad_input;
}

// MSELoss
TORCH_IMPL_FUNC(mse_loss_out_mps)(const Tensor& input, const Tensor& target, int64_t reduction, const Tensor& output_) {
  std::string op_name = "mse_loss_out_mps";
  using namespace mps;
  if ((input.numel() == 0) || (target.numel() == 0)) {
    reduction == Reduction::Mean ? output_.fill_(std::numeric_limits<float>::quiet_NaN()) : output_.zero_();
    return;
  }
  bool contiguousOutput = !needsGather(output_);
  Tensor output = output_;
  if (!contiguousOutput) {
    output = output_.contiguous();
  }

  TORCH_CHECK(target.is_same_size(input), op_name + ": target and input tensors must have identical shapes");
  TORCH_CHECK(c10::isFloatingType(input.scalar_type()) && c10::isFloatingType(target.scalar_type()),
              op_name + ": only defined for floating types");
  TORCH_CHECK(output.is_mps());

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor = nil;
    MPSGraphTensor* targetTensor = nil;
    MPSGraphTensor* outputTensor = nil;
  };

  @autoreleasepool {
    std::string key = op_name + reductionToString(reduction) + getTensorsStringKey({input, target});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      newCachedGraph->targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, target);

      MPSGraphTensor* diffTensor = [mpsGraph subtractionWithPrimaryTensor:newCachedGraph->inputTensor
                                                          secondaryTensor:newCachedGraph->targetTensor
                                                                     name:nil];
      MPSGraphTensor* diffSquareTensor = [mpsGraph squareWithTensor:diffTensor name:nil];
      newCachedGraph->outputTensor = reduceTensor(diffSquareTensor, reduction, mpsGraph, input.sizes().size());
    });
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor, input);
    Placeholder targetPlaceholder = Placeholder(cachedGraph->targetTensor, target);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, contiguousOutput ? output_ : output);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder, targetPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (!contiguousOutput) {
    output_.copy_(output);
  }
}

Tensor& mse_loss_backward_out_mps(const Tensor& grad_output,
                                  const Tensor& input,
                                  const Tensor& target,
                                  int64_t reduction,
                                  Tensor& grad_input) {
  return mps::mse_loss_backward_out_impl(grad_output, input, target, reduction, grad_input, __func__);
}

Tensor mse_loss_backward_mps(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
  Tensor grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::mse_loss_backward_out_impl(grad_output, input, target, reduction, grad_input, __func__);
}

// BCELoss
Tensor& binary_cross_entropy_out_mps(const Tensor& input,
                                     const Tensor& target,
                                     const std::optional<Tensor>& weight_opt,
                                     int64_t reduction,
                                     Tensor& loss) {
  return mps::bce_loss_metal(input, target, weight_opt, reduction, loss, std::nullopt, __func__);
}

Tensor binary_cross_entropy_mps(const Tensor& input,
                                const Tensor& target,
                                const std::optional<Tensor>& weight_opt,
                                int64_t reduction) {
  Tensor loss = at::empty_like(input);
  return mps::bce_loss_metal(input, target, weight_opt, reduction, loss, std::nullopt, __func__);
}

Tensor& binary_cross_entropy_backward_out_mps(const Tensor& grad_output,
                                              const Tensor& input,
                                              const Tensor& target,
                                              const std::optional<Tensor>& weight_opt,
                                              int64_t reduction,
                                              Tensor& grad_input) {
  return mps::bce_loss_metal(input, target, weight_opt, reduction, grad_input, grad_output, __func__);
}

Tensor binary_cross_entropy_backward_mps(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         const std::optional<Tensor>& weight_opt,
                                         int64_t reduction) {
  Tensor grad_input = at::empty_like(input);
  return mps::bce_loss_metal(input, target, weight_opt, reduction, grad_input, grad_output, __func__);
}

// SmoothL1Loss
TORCH_IMPL_FUNC(smooth_l1_loss_out_mps)
(const Tensor& input, const Tensor& target, int64_t reduction, double beta, const Tensor& result) {
  mps::smooth_l1_loss_template(input, target, reduction, beta, result);
}

Tensor& smooth_l1_loss_backward_out_mps(const Tensor& grad_output,
                                        const Tensor& input,
                                        const Tensor& target,
                                        int64_t reduction,
                                        double beta,
                                        Tensor& grad_input) {
  mps::smooth_l1_loss_backward_impl(grad_output, input, target, reduction, beta, grad_input);

  return grad_input;
}

// NLLLoss
TORCH_IMPL_FUNC(nll_loss_backward_out_mps)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input) {
  const Tensor& weight = weight_opt.getTensorRef();

  mps::nllnd_loss_backward_impl(
      (Tensor&)grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight, false);
  return;
}

TORCH_IMPL_FUNC(nll_loss_forward_out_mps)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  const Tensor& weight = weight_opt.getTensorRef();

  mps::nllnd_loss_forward_impl(
      (Tensor&)output, (Tensor&)total_weight, self, target, weight, reduction, ignore_index, false);

  return;
}

inline void check_inputs_nll_loss2d(const Tensor& input, const Tensor& target, const Tensor& weight) {
  TORCH_CHECK(target.dim() == 3,
              "only batches of spatial targets supported (3D tensors)"
              " but got targets of dimension: ",
              target.dim());
  TORCH_CHECK(input.dim() == 4,
              "only batches of spatial inputs supported (4D tensors), "
              "but got input of dimension: ",
              input.dim());
  TORCH_CHECK(!weight.defined() || weight.numel() == input.size(1),
              "weight tensor should be defined either for all or no classes");

  const int64_t input0 = input.size(0);
  const int64_t input2 = input.size(2);
  const int64_t input3 = input.size(3);
  const int64_t target0 = target.size(0);
  const int64_t target1 = target.size(1);
  const int64_t target2 = target.size(2);
  TORCH_CHECK(input0 == target0 && input2 == target1 && input3 == target2,
              "size mismatch (got input: ",
              input.sizes(),
              " , target: ",
              target.sizes());
}

static void nll_loss2d_forward_out_mps_template(Tensor& output,
                                                Tensor& total_weight,
                                                const Tensor& input,
                                                const Tensor& target,
                                                const Tensor& weight,
                                                int64_t reduction,
                                                int64_t ignore_index) {
  check_inputs_nll_loss2d(input, target, weight);
  total_weight.resize_({});

  mps::nllnd_loss_forward_impl(output, total_weight, input, target, weight, reduction, ignore_index, true);

  return;
}

std::tuple<Tensor&, Tensor&> nll_loss2d_forward_out_mps(const Tensor& self,
                                                        const Tensor& target,
                                                        const std::optional<Tensor>& weight_opt,
                                                        int64_t reduction,
                                                        int64_t ignore_index,
                                                        Tensor& output,
                                                        Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  nll_loss2d_forward_out_mps_template(output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_mps(const Tensor& self,
                                                  const Tensor& target,
                                                  const std::optional<Tensor>& weight_opt,
                                                  int64_t reduction,
                                                  int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  at::native::nll_loss2d_forward_out_mps(self, target, weight, reduction, ignore_index, output, total_weight);
  return std::make_tuple(output, total_weight);
}

static void nll_loss2d_backward_out_mps_template(Tensor& grad_input,
                                                 const Tensor& grad_output,
                                                 const Tensor& input,
                                                 const Tensor& target,
                                                 const Tensor& weight,
                                                 int64_t reduction,
                                                 int64_t ignore_index,
                                                 const Tensor& total_weight) {
  check_inputs_nll_loss2d(input, target, weight);
  grad_input.resize_as_(input);
  grad_input.zero_();
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(total_weight.numel() == 1,
              "expected total_weight to be a single element tensor, got: ",
              total_weight.sizes(),
              " (",
              total_weight.numel(),
              " elements)");

  mps::nllnd_loss_backward_impl(
      grad_input, grad_output, input, target, weight, reduction, ignore_index, total_weight, true);

  return;
}

Tensor& nll_loss2d_backward_out_mps(const Tensor& grad_output,
                                    const Tensor& self,
                                    const Tensor& target,
                                    const std::optional<Tensor>& weight_opt,
                                    int64_t reduction,
                                    int64_t ignore_index,
                                    const Tensor& total_weight,
                                    Tensor& grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  nll_loss2d_backward_out_mps_template(
      grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  return grad_input;
}

Tensor nll_loss2d_backward_mps(const Tensor& grad_output,
                               const Tensor& self,
                               const Tensor& target,
                               const std::optional<Tensor>& weight_opt,
                               int64_t reduction,
                               int64_t ignore_index,
                               const Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  auto grad_input = at::zeros_like(self);
  nll_loss2d_backward_out_mps(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  return grad_input;
}

} // namespace at::native
