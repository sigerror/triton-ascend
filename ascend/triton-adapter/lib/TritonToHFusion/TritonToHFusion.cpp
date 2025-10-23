/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#include "TritonToHFusion/Passes.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOHFUSION
#include "ascend/triton-adapter/include/TritonToHFusion/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hfusion;

namespace {
  struct TritonHistogramToHFusionConversion
    : OpRewritePattern<triton::HistogramOp> {
    using OpRewritePattern<triton::HistogramOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(triton::HistogramOp op,
      PatternRewriter &rewriter) const final {
      auto loc = op.getLoc();
      Value input = op.getSrc();
      auto resultType = op.getResult().getType();

      int64_t numBins = 256; // 256 is default fallback.
      if (auto rankedTy = dyn_cast<RankedTensorType>(resultType))
        if (rankedTy.hasStaticShape() && rankedTy.getNumElements() > 0)
          numBins = rankedTy.getNumElements();

      auto numBinsAttr = rewriter.getI64IntegerAttr(numBins);

      auto newOp = rewriter.create<hfusion::HistogramOp>(
        loc, resultType, input, numBinsAttr, Value());

      rewriter.replaceOp(op, newOp.getResult());
      return success();
    }
  };
}

namespace {
    struct TritonToHFusionPass
        : public mlir::triton::impl::TritonToHFusionBase<
            TritonToHFusionPass> {
    void runOnOperation() override;
    };
} // namespace

void TritonToHFusionPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<hfusion::HFusionDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<TritonHistogramToHFusionConversion>(patterns.getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createTritonToHFusionPass() {
  return std::make_unique<TritonToHFusionPass>();
}