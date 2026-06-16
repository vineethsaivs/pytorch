#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/PyProcessGroup.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

class PyBackend : public Backend {
 public:
  using Backend::Backend;

  PyBackend(py::object backend, int rank, int size)
      : Backend(rank, size), pyBackend_(std::move(backend)) {}

  ~PyBackend() override {
    pybind11::gil_scoped_acquire gil;
    if (pyBackend_) {
      pyBackend_.dec_ref();
      pyBackend_.ptr() = nullptr;
    }
  }

  static c10::intrusive_ptr<Backend> wrap(py::object backend) {
    if (backend.is_none()) {
      return nullptr;
    }

    auto base = backend.cast<c10::intrusive_ptr<Backend>>();
    if (dynamic_cast<PyBackend*>(base.get()) == nullptr ||
        !hasPythonBackendClass(backend)) {
      return base;
    }
    return c10::make_intrusive<PyBackend>(
        std::move(backend), base->getRank(), base->getSize());
  }

  py::object getPyBackendAttr(const std::string& name) const {
    if (pyBackend_) {
      return py::getattr(pyBackend_, name.c_str());
    }
    throw py::attribute_error(name);
  }

 private:
  static bool isC10dBackendType(py::handle type) {
    py::object name = py::getattr(type, "__name__", py::none());
    py::object module = py::getattr(type, "__module__", py::none());
    return !name.is_none() && !module.is_none() &&
        name.cast<std::string>() == "Backend" &&
        module.cast<std::string>() == "torch._C._distributed_c10d";
  }

  static bool hasPythonBackendClass(py::handle pySelf) {
    py::tuple mro = py::type::handle_of(pySelf).attr("__mro__");
    for (py::handle typeHandle : mro) {
      if (isC10dBackendType(typeHandle)) {
        return false;
      }
      py::object module = py::getattr(typeHandle, "__module__", py::none());
      if (!module.is_none() &&
          module.cast<std::string>() != "torch._C._distributed_c10d") {
        return true;
      }
    }
    return false;
  }

  py::object pySelf() const {
    if (pyBackend_) {
      return pyBackend_;
    }
    return py::cast(
        const_cast<PyBackend*>(this), py::return_value_policy::reference);
  }

  std::optional<py::object> getAttrOverride(const char* name) const {
    py::object self = pySelf();
    py::tuple mro = py::type::handle_of(self).attr("__mro__");
    for (py::handle typeHandle : mro) {
      if (isC10dBackendType(typeHandle)) {
        return std::nullopt;
      }
      py::object dict = py::getattr(typeHandle, "__dict__", py::none());
      if (!dict.is_none()) {
        int hasKey = PyMapping_HasKeyString(dict.ptr(), name);
        if (hasKey == -1) {
          throw py::error_already_set();
        }
        if (hasKey == 1) {
          return py::getattr(self, name);
        }
      }
    }
    return std::nullopt;
  }

  static c10::intrusive_ptr<Work> wrapWork(py::object work) {
    return c10::make_intrusive<PyProcessGroup::PyWorkHolder>(std::move(work));
  }

#define BACKEND_ATTR_OVERRIDE(py_name, return_type, base_call) \
  do {                                                         \
    pybind11::gil_scoped_acquire gil;                          \
    if (auto override = getAttrOverride(py_name)) {            \
      return override->cast<return_type>();                    \
    }                                                          \
    return base_call;                                          \
  } while (false)

#define BACKEND_RETURN_OVERRIDE0(py_name, return_type, base_call) \
  do {                                                            \
    pybind11::gil_scoped_acquire gil;                             \
    if (auto override = getAttrOverride(py_name)) {               \
      return (*override)().cast<return_type>();                   \
    }                                                             \
    return base_call;                                             \
  } while (false)

#define BACKEND_RETURN_OVERRIDE(py_name, return_type, base_call, ...) \
  do {                                                                \
    pybind11::gil_scoped_acquire gil;                                 \
    if (auto override = getAttrOverride(py_name)) {                   \
      return (*override)(__VA_ARGS__).cast<return_type>();            \
    }                                                                 \
    return base_call;                                                 \
  } while (false)

#define BACKEND_VOID_OVERRIDE0(py_name, base_call)  \
  do {                                              \
    pybind11::gil_scoped_acquire gil;               \
    if (auto override = getAttrOverride(py_name)) { \
      (*override)();                                \
      return;                                       \
    }                                               \
    return base_call;                               \
  } while (false)

#define BACKEND_VOID_OVERRIDE(py_name, base_call, ...) \
  do {                                                 \
    pybind11::gil_scoped_acquire gil;                  \
    if (auto override = getAttrOverride(py_name)) {    \
      (*override)(__VA_ARGS__);                        \
      return;                                          \
    }                                                  \
    return base_call;                                  \
  } while (false)

#define BACKEND_BACKEND_OVERRIDE(py_name, base_call, ...) \
  do {                                                    \
    pybind11::gil_scoped_acquire gil;                     \
    if (auto override = getAttrOverride(py_name)) {       \
      return wrap((*override)(__VA_ARGS__));              \
    }                                                     \
    return base_call;                                     \
  } while (false)

#define BACKEND_NAMED_WORK_OVERRIDE0(py_name, base_call) \
  do {                                                   \
    pybind11::gil_scoped_acquire gil;                    \
    if (auto override = getAttrOverride(py_name)) {      \
      return wrapWork((*override)());                    \
    }                                                    \
    return base_call;                                    \
  } while (false)

#define BACKEND_NAMED_WORK_OVERRIDE(py_name, base_call, ...) \
  do {                                                       \
    pybind11::gil_scoped_acquire gil;                        \
    if (auto override = getAttrOverride(py_name)) {          \
      return wrapWork((*override)(__VA_ARGS__));             \
    }                                                        \
    return base_call;                                        \
  } while (false)

#define BACKEND_WORK_OVERRIDE(name, ...) \
  BACKEND_NAMED_WORK_OVERRIDE(#name, Backend::name(__VA_ARGS__), __VA_ARGS__)

#define BACKEND_BOOL_PROPERTY_OVERRIDE(cname, py_name)      \
  bool cname() const override {                             \
    BACKEND_ATTR_OVERRIDE(py_name, bool, Backend::cname()); \
  }

  py::object pyBackend_;
  using AllocatorPtr = std::shared_ptr<c10::Allocator>;
  using OptionsPtr = c10::intrusive_ptr<Options>;
  using WindowPtr = c10::intrusive_ptr<Window>;
  using MemoryStats = std::unordered_map<std::string, uint64_t>;

 public:
  BACKEND_BOOL_PROPERTY_OVERRIDE(supportsSplitting, "supports_splitting")

  BACKEND_BOOL_PROPERTY_OVERRIDE(supportsCoalescing, "supports_coalescing")

  BACKEND_BOOL_PROPERTY_OVERRIDE(
      supportsTimeEstimation,
      "supports_time_estimate")

  BACKEND_BOOL_PROPERTY_OVERRIDE(supportsShrinking, "supports_shrinking")

  c10::intrusive_ptr<Backend> shrink(
      const std::vector<int64_t>& ranks_to_exclude,
      int shrink_flags = 0,
      const c10::intrusive_ptr<Options>& opts_override = nullptr) override {
    BACKEND_BACKEND_OVERRIDE(
        "shrink",
        Backend::shrink(ranks_to_exclude, shrink_flags, opts_override),
        ranks_to_exclude,
        shrink_flags,
        opts_override);
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    BACKEND_VOID_OVERRIDE("set_timeout", Backend::setTimeout(timeout), timeout);
  }

  BACKEND_BOOL_PROPERTY_OVERRIDE(supportsReconfigure, "supports_reconfigure")

  ReconfigureHandle get_reconfigure_handle() const override {
    BACKEND_RETURN_OVERRIDE0(
        "get_reconfigure_handle",
        ReconfigureHandle,
        Backend::get_reconfigure_handle());
  }

  c10::intrusive_ptr<Work> reconfigure(
      const ReconfigureOptions& opts) override {
    BACKEND_WORK_OVERRIDE(reconfigure, opts);
  }

  BACKEND_BOOL_PROPERTY_OVERRIDE(supportsWindow, "supports_window")

  c10::intrusive_ptr<Window> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override {
    BACKEND_RETURN_OVERRIDE(
        "new_window", WindowPtr, Backend::new_window(tensor), tensor);
  }

  void startCoalescing() override {
    BACKEND_VOID_OVERRIDE0("start_coalescing", Backend::startCoalescing());
  }

  c10::intrusive_ptr<Work> endCoalescing() override {
    BACKEND_NAMED_WORK_OVERRIDE0("end_coalescing", Backend::endCoalescing());
  }

  const std::string getBackendName() const override {
    BACKEND_RETURN_OVERRIDE0(
        "getBackendName", std::string, Backend::getBackendName());
  }

  c10::intrusive_ptr<Options> getBackendOptions() override {
    BACKEND_ATTR_OVERRIDE("options", OptionsPtr, Backend::getBackendOptions());
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    BACKEND_WORK_OVERRIDE(broadcast, tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    BACKEND_WORK_OVERRIDE(allreduce, tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    BACKEND_WORK_OVERRIDE(allreduce_sparse, tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    BACKEND_WORK_OVERRIDE(allreduce_coalesced, tensors, opts);
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override {
    BACKEND_WORK_OVERRIDE(reduce, tensors, opts);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    BACKEND_WORK_OVERRIDE(allgather, outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    BACKEND_WORK_OVERRIDE(all_gather_single, outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    BACKEND_WORK_OVERRIDE(
        allgather_coalesced, outputTensorLists, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> all_gather_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    BACKEND_WORK_OVERRIDE(all_gather_single_coalesced, outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override {
    BACKEND_WORK_OVERRIDE(gather, outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override {
    BACKEND_WORK_OVERRIDE(scatter, outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    BACKEND_WORK_OVERRIDE(reduce_scatter, outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    BACKEND_WORK_OVERRIDE(
        reduce_scatter_single, outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    BACKEND_WORK_OVERRIDE(
        reduce_scatter_single_coalesced, outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> all_to_all_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    BACKEND_WORK_OVERRIDE(
        all_to_all_single,
        outputBuffer,
        inputBuffer,
        outputSplitSizes,
        inputSplitSizes,
        opts);
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    BACKEND_WORK_OVERRIDE(alltoall, outputTensors, inputTensors, opts);
  }

  void monitoredBarrier(const BarrierOptions& opts, bool waitAllRanks = false)
      override {
    BACKEND_VOID_OVERRIDE(
        "monitored_barrier",
        Backend::monitoredBarrier(opts, waitAllRanks),
        opts,
        waitAllRanks);
  }

  void setSequenceNumberForGroup() override {
    BACKEND_VOID_OVERRIDE0(
        "_set_sequence_number_for_group", Backend::setSequenceNumberForGroup());
  }

  uint64_t getSequenceNumberForGroup() override {
    BACKEND_RETURN_OVERRIDE0(
        "_get_sequence_number_for_group",
        uint64_t,
        Backend::getSequenceNumberForGroup());
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    BACKEND_WORK_OVERRIDE(send, tensors, dstRank, tag);
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    BACKEND_WORK_OVERRIDE(recv, tensors, srcRank, tag);
  }

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override {
    BACKEND_NAMED_WORK_OVERRIDE(
        "recv_anysource", Backend::recvAnysource(tensors, tag), tensors, tag);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    BACKEND_WORK_OVERRIDE(barrier, opts);
  }

  void registerOnCompletionHook(
      std::function<void(std::shared_ptr<WorkInfo>)>&& hook) override {
    pybind11::gil_scoped_acquire gil;
    auto hookCopy = hook;
    onCompletionHook_ = std::move(hook);
    if (auto override = getAttrOverride("_register_on_completion_hook")) {
      (*override)(std::move(hookCopy));
    }
  }

  void waitForPendingWorks() override {
    BACKEND_VOID_OVERRIDE0(
        "_wait_for_pending_works", Backend::waitForPendingWorks());
  }

  void enableCollectivesTiming() override {
    BACKEND_VOID_OVERRIDE0(
        "_enable_collectives_timing", Backend::enableCollectivesTiming());
  }

  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<Options>& opts) override {
    BACKEND_BACKEND_OVERRIDE(
        "split", Backend::split(store, ranks, opts), store, ranks, opts);
  }

  c10::intrusive_ptr<Backend> merge(
      const c10::intrusive_ptr<Store>& store,
      const c10::intrusive_ptr<Options>& opts,
      const int& rank,
      const int& size) override {
    BACKEND_BACKEND_OVERRIDE(
        "merge",
        Backend::merge(store, opts, rank, size),
        store,
        opts,
        rank,
        size);
  }

  void setGroupUid(const std::string& pg_uid) override {
    BACKEND_VOID_OVERRIDE(
        "_set_group_name", Backend::setGroupUid(pg_uid), pg_uid);
  }

  void eagerConnectSingleDevice(at::Device device) override {
    BACKEND_VOID_OVERRIDE(
        "eager_connect_single_device",
        Backend::eagerConnectSingleDevice(device),
        device);
  }

  ErrorType getError() override {
    BACKEND_RETURN_OVERRIDE0("get_error", ErrorType, Backend::getError());
  }

  std::shared_ptr<c10::Allocator> getMemAllocator() override {
    BACKEND_ATTR_OVERRIDE(
        "mem_allocator", AllocatorPtr, Backend::getMemAllocator());
  }

  at::Tensor allocateTensor(long size, at::TensorOptions options = {})
      override {
    pybind11::gil_scoped_acquire gil;
    if (auto override = getAttrOverride("allocate_tensor")) {
      c10::ScalarType dtype = c10::optTypeMetaToScalarType(options.dtype_opt())
                                  .value_or(at::kFloat);
      c10::Device device =
          options.device_opt().value_or(c10::Device(c10::DeviceType::CPU));
      return (*override)(size, dtype, device).cast<at::Tensor>();
    }
    return Backend::allocateTensor(size, options);
  }

  bool supportsTensorAlloc(c10::DeviceIndex deviceIdx) override {
    BACKEND_RETURN_OVERRIDE(
        "supports_tensor_alloc",
        bool,
        Backend::supportsTensorAlloc(deviceIdx),
        deviceIdx);
  }

  void abort() override {
    BACKEND_VOID_OVERRIDE0("abort", Backend::abort());
  }

  void shutdown() override {
    BACKEND_VOID_OVERRIDE0("shutdown", Backend::shutdown());
  }

  void suspend() override {
    BACKEND_VOID_OVERRIDE0("suspend", Backend::suspend());
  }

  void resume() override {
    BACKEND_VOID_OVERRIDE0("resume", Backend::resume());
  }

  std::unordered_map<std::string, uint64_t> getMemoryStats() override {
    BACKEND_RETURN_OVERRIDE0(
        "memory_stats", MemoryStats, Backend::getMemoryStats());
  }

#undef BACKEND_BOOL_PROPERTY_OVERRIDE
#undef BACKEND_WORK_OVERRIDE
#undef BACKEND_NAMED_WORK_OVERRIDE
#undef BACKEND_NAMED_WORK_OVERRIDE0
#undef BACKEND_BACKEND_OVERRIDE
#undef BACKEND_VOID_OVERRIDE
#undef BACKEND_VOID_OVERRIDE0
#undef BACKEND_RETURN_OVERRIDE
#undef BACKEND_RETURN_OVERRIDE0
#undef BACKEND_ATTR_OVERRIDE
};

} // namespace c10d
