#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>

#define DEBUG_TYPE "triton-unstructure-converter"

using namespace mlir;
using namespace triton;

#include "llvm/Support/Debug.h"

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::extractOp(
    Location loc, Value value, ArrayRef<Value> iterIdx,
    PatternRewriter &rewriter) const {
  if (!value)
    return value;
  auto extractedOp = rewriter.create<tensor::ExtractOp>(loc, value, iterIdx);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <typename MemAccOpTy>
UnstructuredMemAccessConverter<MemAccOpTy>::UnstructuredMemAccessConverter(
    MLIRContext *context, const llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap)
    : OpRewritePattern<MemAccOpTy>(context), offsetMap(offsetMap) {}

template <typename MemAccOpTy>
LogicalResult UnstructuredMemAccessConverter<MemAccOpTy>::matchAndRewrite(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto ptr = op.getPtr();
  auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());

  if (!ptrType || ptrType.getRank() == 1 && ptrType.getShape()[0] == 1) {
    return failure();
  }

  if (!offsetMap.contains(ptr))
    return op.emitError() << "PtrOffsetInfo should be computed\n" << ptr;

  auto ptrOffsetInfo = offsetMap.at(ptr);

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Converting " << op->getName() << "\n";
    os << op << "\n";
    os << ptrOffsetInfo.isStructured() << "\n";
  });

  if (ptrOffsetInfo.isStructured()) {
    return failure();
  }

  auto srcPtr = ptrOffsetInfo.getPtr();
  auto offset = ptrOffsetInfo.getOffset();

  // LoadLike is operation with result
  constexpr bool isLoadLike = std::is_same_v<MemAccOpTy, triton::LoadOp> ||
                              std::is_same_v<MemAccOpTy, triton::AtomicRMWOp> ||
                              std::is_same_v<MemAccOpTy, triton::AtomicCASOp>;

  // Check if offsetShape is equal to localMem shape

  Value zeroIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto resultShape = ptrType.getShape();
  auto resultElementType =
      cast<triton::PointerType>(ptrType.getElementType()).getPointeeType();

  Value iterArg = nullptr;

  // Only load case
  if constexpr (isLoadLike) {
    iterArg =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, resultElementType);
  }
  Value newOpResult = nullptr;

  auto insertPoint = rewriter.saveInsertionPoint();

  SmallVector<OpFoldResult> dims(resultShape.size(), rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> strides;
  SmallVector<Value> iterIdx;

  SmallVector<int64_t> localMemStrides(1, 1);

  for (auto size : llvm::reverse(resultShape)) {
    localMemStrides.push_back(localMemStrides.back() * size);
  }
  localMemStrides.pop_back();

  std::reverse(localMemStrides.begin(), localMemStrides.end());
  bool isExtractedAttrInserted = false;
  for (const auto &[size, localMemStride] :
       llvm::zip_equal(resultShape, localMemStrides)) {
    // handle indirect dimension
    strides.push_back(rewriter.getIndexAttr(localMemStride));
    Value sizeVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(size));
    scf::ForOp forOp;
    if constexpr (isLoadLike) {
      forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, sizeVal, oneIdx,
                                          ValueRange({iterArg}));
      if (!newOpResult) {
        newOpResult = forOp->getResult(0);
      } else {
        rewriter.create<scf::YieldOp>(loc, forOp->getResult(0));
      }
      iterArg = forOp.getRegionIterArg(0);
    } else {
      forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, sizeVal, oneIdx);
    }
    offsets.push_back(forOp.getInductionVar());
    iterIdx.push_back(forOp.getInductionVar());
    if (!isExtractedAttrInserted) {
      forOp->setAttr("ExtractedLoadOrStore", UnitAttr::get(rewriter.getContext()));
      isExtractedAttrInserted=true;
    }
    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  auto scalarLikeShape = SmallVector<int64_t>(dims.size(), 1);
  auto scalarLikeType =
      RankedTensorType::get(scalarLikeShape, resultElementType);

  auto offsetType = cast<RankedTensorType>(offset.getType());
  auto extractedOffset = extractOp(loc, offset, iterIdx, rewriter);
  if (isa<RankedTensorType>(srcPtr.getType())) {
    srcPtr = extractOp(loc, srcPtr, iterIdx, rewriter);
  }
  Value ptrToAccess = rewriter.create<triton::AddPtrOp>(
      loc, srcPtr.getType(), srcPtr, extractedOffset);
  if constexpr (isLoadLike) {
    Value accessedValue;
    assert(iterArg && "Load case must have iterArg in for loop");
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
      auto extractedMask = extractOp(loc, op.getMask(), iterIdx, rewriter);
      auto extractedOther = extractOp(loc, op.getOther(), iterIdx, rewriter);
      accessedValue = rewriter.create<triton::LoadOp>(
          loc, ptrToAccess, extractedMask, extractedOther,
          /*boundaryCheck=*/ArrayRef<int32_t>(), /*PaddingOptionAttr=*/nullptr);
    } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
      auto extractedValue = extractOp(loc, op.getVal(), iterIdx, rewriter);
      Value extractedMask = extractOp(loc, op.getMask(), iterIdx, rewriter);
      auto splatedPtrToAccess = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
          ptrToAccess);
      auto splatedExtractedValue = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
          extractedValue);
      auto splatedExtractedMask = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, extractedMask.getType()),
          extractedMask);
      accessedValue = rewriter.create<triton::AtomicRMWOp>(
          loc, scalarLikeType, op.getAtomicRmwOpAttr(), splatedPtrToAccess,
          splatedExtractedValue, splatedExtractedMask, op.getSemAttr(),
          op.getScopeAttr());
    } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicCASOp>) {
      auto extractedCmp = extractOp(loc, op.getCmp(), iterIdx, rewriter);
      auto extractedValue = extractOp(loc, op.getVal(), iterIdx, rewriter);
      auto splatedPtrToAccess = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
          ptrToAccess);
      auto splatedExtractedCmp = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, extractedCmp.getType()),
          extractedCmp);
      auto splatedExtractedValue = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
          extractedValue);
      accessedValue = rewriter.create<triton::AtomicCASOp>(
          loc, scalarLikeType, splatedPtrToAccess, splatedExtractedCmp,
          splatedExtractedValue, op.getSemAttr(), op.getScopeAttr());
    }
    accessedValue.getDefiningOp()->setAttr(
        ConverterUtils::discreteAttrName, UnitAttr::get(rewriter.getContext()));

    Value splatedValue = accessedValue;
    if (!isa<RankedTensorType>(splatedValue.getType())) {
      splatedValue =
          rewriter.create<triton::SplatOp>(loc, scalarLikeType, accessedValue);
    }
    auto result = rewriter.create<tensor::InsertSliceOp>(
        loc, splatedValue, iterArg, offsets, dims, strides);
    rewriter.create<scf::YieldOp>(loc, result->getResult(0));

    result->setAttr(ConverterUtils::discreteAttrName,
                    UnitAttr::get(rewriter.getContext()));
    rewriter.restoreInsertionPoint(insertPoint);
    rewriter.replaceOp(op, newOpResult);
  } else if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
    auto extractedValue = extractOp(loc, op.getValue(), iterIdx, rewriter);
    auto extractedMask = extractOp(loc, op.getMask(), iterIdx, rewriter);
    if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
      rewriter.create<triton::StoreOp>(loc, ptrToAccess, extractedValue,
                                       extractedMask);
    }
    rewriter.eraseOp(op);
  } else {
    llvm_unreachable("Unhandled discrete memory access operation");
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "After conversion\n"
       << ptrToAccess.getDefiningOp()
              ->template getParentOfType<triton::FuncOp>()
       << "\n";
  });
  return success();
}

template <typename LoopOpTy, typename>
void TritonToUnstructurePass::runPreparse(LoopOpTy op) {
  IRRewriter rewriter(&getContext());
  for (const auto &[idx, res]: llvm::enumerate(op->getResults())) {
    if (isa<RankedTensorType>(res.getType())) {
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Pre-parsing " << op->getName() << "\n" << op << "\n";
      });
      parse(res, op.getLoc(), rewriter, offsetMapForLoopArgs);

      Value regionIterArg;
      if constexpr (std::is_same_v<LoopOpTy, scf::ForOp>) {
        regionIterArg = op.getRegionIterArg(idx);
      } else if constexpr (std::is_same_v<LoopOpTy, scf::WhileOp>) {
        regionIterArg = op.getRegionIterArgs()[idx];
      }
      offsetMap[regionIterArg] = PtrOffsetInfo();
      SmallVector<bool> &regionIterArgOffset =
          offsetMap[regionIterArg].getStructuredRef();
      SmallVector<bool> &resOffset =
          offsetMapForLoopArgs[res].getStructuredRef();
      regionIterArgOffset.resize(resOffset.size());
      for (size_t i = 0; i < resOffset.size(); i++)
        regionIterArgOffset[i] = resOffset[i];
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Pre-parsing result of\n" << regionIterArg << "\nis ";
        for (size_t i = 0; i < resOffset.size(); i++)
          os << resOffset[i];
        os << '\n';
      });
    }
  }
}

template <typename MemAccOpTy, typename>
void TritonToUnstructurePass::runParse(MemAccOpTy op) {
  IRRewriter rewriter(&getContext());
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing " << op->getName() << "\n" << op << "\n";
  });
  parse(op.getPtr(), op.getLoc(), rewriter, offsetMap);
}

void TritonToUnstructurePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  moduleOp->walk([this](scf::ForOp op) { runPreparse(op); });
  moduleOp->walk([this](scf::WhileOp op) { runPreparse(op); });

  moduleOp->walk([this](triton::LoadOp op) { runParse(op); });
  moduleOp->walk([this](triton::StoreOp op) { runParse(op); });
  moduleOp->walk([this](triton::AtomicRMWOp op) { runParse(op); });
  moduleOp->walk([this](triton::AtomicCASOp op) { runParse(op); });

  RewritePatternSet patterns(ctx);

  patterns.add<UnstructuredMemAccessConverter<triton::LoadOp>,
               UnstructuredMemAccessConverter<triton::StoreOp>,
               UnstructuredMemAccessConverter<triton::AtomicRMWOp>,
               UnstructuredMemAccessConverter<triton::AtomicCASOp>>(ctx,
                                                                    offsetMap);

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing done\n";
  });

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Patterns");
    signalPassFailure();
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }
}

void TritonToUnstructurePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect,
                  affine::AffineDialect, scf::SCFDialect, tensor::TensorDialect,
                  bufferization::BufferizationDialect, memref::MemRefDialect,
                  triton::TritonDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToUnstructurePass() {
  return std::make_unique<TritonToUnstructurePass>();
}