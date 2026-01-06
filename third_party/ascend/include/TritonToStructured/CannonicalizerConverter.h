/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef TRITON_ADAPTER_CANNONICALIZERCONVERTER_H
#define TRITON_ADAPTER_CANNONICALIZERCONVERTER_H

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace CannonicalizerConverter {

using namespace mlir;
using namespace triton;

void rewriteLoopOp(LoopLikeOpInterface op, ConversionPatternRewriter& rewriter);

template <typename LoopOpTy,
          typename = std::enable_if_t<std::is_same_v<LoopOpTy, scf::ForOp> ||
                                      std::is_same_v<LoopOpTy, scf::WhileOp>>>
class LoopConverter : public OpConversionPattern<LoopOpTy> {
public:
    using OpConversionPattern<LoopOpTy>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        LoopOpTy op,
        typename OpConversionPattern<LoopOpTy>::OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter
    ) const override {
        CannonicalizerConverter::rewriteLoopOp(op, rewriter);
        return failure();
    }
};

}  // namespace CannonicalizerConverter

#endif