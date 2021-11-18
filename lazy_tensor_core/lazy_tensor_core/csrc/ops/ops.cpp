#include "lazy_tensor_core/csrc/ops/ops.h"

#include <c10/util/Half.h>

#include <cmath>

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyLazyIr.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "torch/csrc/lazy/core/ir_metadata.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

#define PTLTC_BINARY_OP(name, sym)                                           \
  NodePtr name(const torch::lazy::Value& input0, const torch::lazy::Value& input1) {                   \
    NodePtr node = GenericOp(OpKind(sym), {input0, input1});                 \
    std::dynamic_pointer_cast<TsNode>(node)->SetShapeDeferred(                                                  \
        [&]() { return compiler::InferShape(node.get()); }); \
    return node;                                                             \
  }

PTLTC_BINARY_OP(Pow, at::aten::pow);

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
