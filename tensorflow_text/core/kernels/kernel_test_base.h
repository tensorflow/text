// Copyright 2026 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_KERNEL_TEST_BASE_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_KERNEL_TEST_BASE_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/version.h"

#ifndef TF_ASSERT_OK
#define TF_ASSERT_OK(expr) ASSERT_TRUE((expr).ok())
#endif

#ifndef TF_EXPECT_OK
#define TF_EXPECT_OK(expr) EXPECT_TRUE((expr).ok())
#endif

namespace tensorflow {

inline FakeInputFunctor FakeInput() {
  return [](const OpDef& op_def, int in_index, const NodeDef& node_def,
            NodeDefBuilder* builder) -> absl::Status {
    char c = 'a' + (in_index % 26);
    std::string in_node(&c, 1);
    DataType dt = op_def.input_arg(in_index).type();
    if (dt == DT_INVALID && !op_def.input_arg(in_index).type_attr().empty()) {
      const std::string& attr_name = op_def.input_arg(in_index).type_attr();
      if (!GetNodeAttr(node_def, attr_name, &dt).ok()) {
        const OpDef::AttrDef* attr = FindAttr(attr_name, op_def);
        if (attr && attr->has_default_value()) {
          dt = attr->default_value().type();
        }
      }
    }
    if (dt == DT_INVALID) {
      dt = DT_INT64;
    }
    builder->Input(in_node, 0, dt);
    return absl::OkStatus();
  };
}

inline FakeInputFunctor FakeInput(DataType dt) {
  return [dt](const OpDef& op_def, int in_index, const NodeDef& node_def,
              NodeDefBuilder* builder) -> absl::Status {
    char c = 'a' + (in_index % 26);
    std::string in_node(&c, 1);
    builder->Input(in_node, 0, dt);
    return absl::OkStatus();
  };
}

namespace text {

// A lightweight CPU device implementing tensorflow::DeviceBase to satisfy
// OpKernelContext without dragging in the heavyweight ThreadPoolDevice from
// //tensorflow/core/common_runtime.
class TestCpuDevice : public DeviceBase {
 public:
  explicit TestCpuDevice(Env* env) : DeviceBase(env) {
    attr_.set_device_type(DEVICE_CPU);
  }
  Allocator* GetAllocator(AllocatorAttributes) override {
    return cpu_allocator();
  }
  const DeviceAttributes& attributes() const override { return attr_; }

 private:
  DeviceAttributes attr_;
};

// Drop-in replacement for OpsTestBase that tests real OpKernels using
// public framework headers and libtensorflow_framework.so symbols.
class KernelTestBase : public ::testing::Test {
 protected:
  KernelTestBase() : device_(Env::Default()), node_def_() {}

  NodeDef* node_def() { return &node_def_; }

  absl::Status InitOp() {
    for (auto* t : inputs_) delete t;
    inputs_.clear();
    input_values_.clear();
    std::shared_ptr<const NodeProperties> props;
    if (absl::Status s = NodeProperties::CreateFromNodeDef(
            node_def_, OpRegistry::Global(), &props);
        !s.ok()) {
      return s;
    }
    OpKernel* kernel = nullptr;
    if (absl::Status s = CreateOpKernel(
            DEVICE_CPU, &device_, device_.GetAllocator({}), /*flib=*/nullptr,
            /*resource_mgr=*/nullptr, props, TF_GRAPH_DEF_VERSION, &kernel);
        !s.ok()) {
      return s;
    }
    kernel_.reset(kernel);
    num_outputs_ = kernel_->num_outputs();
    out_alloc_attrs_.assign(num_outputs_, AllocatorAttributes());
    return absl::OkStatus();
  }

  template <typename T>
  void AddInputFromArray(const TensorShape& shape, absl::Span<const T> data) {
    auto* tensor =
        new Tensor(device_.GetAllocator({}), DataTypeToEnum<T>::v(), shape);
    std::copy(data.begin(), data.end(), tensor->flat<T>().data());
    inputs_.push_back(tensor);
    input_values_.push_back({nullptr, tensor});
  }

  absl::Status RunOpKernel() {
    params_.device = &device_;
    params_.op_kernel = kernel_.get();
    params_.inputs = input_values_;
    params_.output_attr_array = out_alloc_attrs_.data();
    params_.step_container = nullptr;
    params_.cancellation_manager = nullptr;
    params_.stats_collector = nullptr;
    static auto* const default_runner =
        new std::function<void(std::function<void()>)>(
            [](const std::function<void()>& f) { f(); });
    params_.runner = default_runner;

    context_ = std::make_unique<OpKernelContext>(&params_, num_outputs_);
    kernel_->Compute(context_.get());
    return context_->status();
  }

  Tensor* GetOutput(int index) { return context_->mutable_output(index); }

  ~KernelTestBase() override {
    for (auto* t : inputs_) delete t;
  }

  TestCpuDevice device_;
  NodeDef node_def_;
  std::unique_ptr<OpKernel> kernel_;
  int num_outputs_ = 0;
  std::vector<AllocatorAttributes> out_alloc_attrs_;
  std::vector<Tensor*> inputs_;
  std::vector<TensorValue> input_values_;
  OpKernelContext::Params params_;
  std::unique_ptr<OpKernelContext> context_;
};

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_KERNEL_TEST_BASE_H_
