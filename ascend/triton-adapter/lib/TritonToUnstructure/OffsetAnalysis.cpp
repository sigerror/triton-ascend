#include "TritonToUnstructure/OffsetAnalysis.h"
#include "Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-offset-analysis"

namespace mlir {
namespace triton {

PtrOffsetInfo::PtrOffsetInfo(Value ptr, Value offset) : ptr(ptr), offset(offset) {}

Value PtrOffsetInfo::getPtr() const { return this->ptr; }
Value PtrOffsetInfo::getOffset() const { return this->offset; }
bool PtrOffsetInfo::isScalarLike() const { return this->scalarLike; }

SmallVector<bool> &PtrOffsetInfo::getStructuredRef() { return this->structured; }
const SmallVector<bool> &PtrOffsetInfo::getStructured() const {
  return this->structured;
}

void PtrOffsetInfo::setPtr(const Value &ptr) { this->ptr = ptr; }
void PtrOffsetInfo::setOffset(const Value &offset) { this->offset = offset; }

void PtrOffsetInfo::setUnstructured(int rank) {
  this->structured.clear();
  this->structured.resize(rank, false);
}

void PtrOffsetInfo::setStructured(int rank) {
  this->structured.clear();
  this->structured.resize(rank, true);
}

void PtrOffsetInfo::setScalarLike(bool scalarLike) {
  this->scalarLike = scalarLike;
}

bool PtrOffsetInfo::isStructured(int dim) const {
  return this->scalarLike || structured[dim];
}

bool PtrOffsetInfo::isStructured() const {
  return this->scalarLike ||
         llvm::all_of(structured, [](auto dim) { return dim; });
}

bool PtrOffsetInfo::isUnstructured() const {
  return llvm::all_of(structured, [](auto dim) { return !dim; });
}

void parse(Value operand, const Location &loc, RewriterBase &rewriter,
           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  if (offsetMap.contains(operand)) {
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "found\n" << operand << '\n';
    });
    return;
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "parse\n" << operand << '\n';
  });

  if (auto *defOp = operand.getDefiningOp()) {
    if (isa<arith::ArithDialect>(defOp->getDialect())) {
      parseArithOp(defOp, loc, rewriter, offsetMap);
    } else if (isa<triton::TritonDialect>(defOp->getDialect())) {
      parseTritonOp(defOp, loc, rewriter, offsetMap);
    } else {
      if (auto floorOp = dyn_cast<math::FloorOp>(defOp)) {
        parseFloor(floorOp, loc, rewriter, offsetMap);
      } else if (auto ceilOp = dyn_cast<math::CeilOp>(defOp)) {
        parseCeil(ceilOp, loc, rewriter, offsetMap);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
        parseIf(ifOp, loc, rewriter, offsetMap, operand);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(defOp)) {
        parseYield(yieldOp, loc, rewriter, offsetMap);
      } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
        parseFor(forOp, loc, rewriter, offsetMap, operand);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
        parseWhile(whileOp, loc, rewriter, offsetMap, operand);
      } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(defOp)) {
        parseExtract(extractOp, loc, rewriter, offsetMap);
      }
    } 
  } else if (auto ptrType = dyn_cast<triton::PointerType>(operand.getType())) {
    PtrOffsetInfo ptrOffsetInfo;
    ptrOffsetInfo.setPtr(operand);
    auto innerType = ptrType.getPointeeType();

    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(operand);
    if (auto tensorType = dyn_cast<RankedTensorType>(innerType)) {
      Value offset = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(
                   RankedTensorType::get(tensorType.getShape(),
                                         rewriter.getIntegerType(64)),
                   rewriter.getZeroAttr(rewriter.getIntegerType(64))));
      ptrOffsetInfo.setOffset(offset);
      ptrOffsetInfo.getStructuredRef().resize(tensorType.getRank(), true);
    } else {
      ptrOffsetInfo.setOffset(rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(0)));
    }

    offsetMap[operand] = ptrOffsetInfo;
  } else if (auto blockArgument = dyn_cast<BlockArgument>(operand)) {
    auto parentOp = operand.getParentBlock()->getParentOp();
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Handling block argument\n" << *parentOp << '\n';
    });
    if (isa<FunctionOpInterface>(parentOp)) {
      offsetMap[operand] = PtrOffsetInfo();
    } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      auto argNum = blockArgument.getArgNumber();
      offsetMap[operand] = PtrOffsetInfo();
      if (argNum != 0) {
        if (isa<triton::PointerType>(operand.getType())) {
          offsetMap[operand].setPtr(operand);
          offsetMap[operand].setOffset(rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(0)));
        } else if (auto tensorType =
                       dyn_cast<RankedTensorType>(operand.getType());
                   (tensorType &&
                    isa<triton::PointerType>(tensorType.getElementType()))) {
          offsetMap[operand].setPtr(operand);
          RewriterBase::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(forOp.getBody());
          Value offset = rewriter.create<arith::ConstantOp>(
              loc, DenseElementsAttr::get(
                       RankedTensorType::get(tensorType.getShape(),
                                             rewriter.getIntegerType(64)),
                       rewriter.getZeroAttr(rewriter.getIntegerType(64))));
          offsetMap[operand].setOffset(offset);
        }
        Value initArg = forOp.getInitArgs()[argNum - 1];
        parse(initArg, forOp.getLoc(), rewriter, offsetMap);
        SmallVector<bool> &blockArgStructured =
            offsetMap[operand].getStructuredRef();
        SmallVector<bool>& initArgStructured = offsetMap[initArg].getStructuredRef();
        blockArgStructured.resize(initArgStructured.size());
        for (size_t i = 0; i < initArgStructured.size(); i++)
          blockArgStructured[i] = initArgStructured[i];
      }
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      auto argNum = blockArgument.getArgNumber();
      offsetMap[operand] = PtrOffsetInfo();
      if (isa<triton::PointerType>(operand.getType())) {
        offsetMap[operand].setPtr(operand);
        offsetMap[operand].setOffset(rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(0)));
      } else if (auto tensorType =
                     dyn_cast<RankedTensorType>(operand.getType());
                 (tensorType &&
                  isa<triton::PointerType>(tensorType.getElementType()))) {
        offsetMap[operand].setPtr(operand);
        Value offset = rewriter.create<arith::ConstantOp>(
            loc, DenseElementsAttr::get(
                     RankedTensorType::get(tensorType.getShape(),
                                           rewriter.getIntegerType(64)),
                     rewriter.getZeroAttr(rewriter.getIntegerType(64))));
        offsetMap[operand].setOffset(offset);
      }
      Value initArg = whileOp.getInits()[argNum];
      parse(initArg, whileOp.getLoc(), rewriter, offsetMap);
      SmallVector<bool> &blockArgStructured =
          offsetMap[operand].getStructuredRef();
      SmallVector<bool>& initArgStructured = offsetMap[initArg].getStructuredRef();
      blockArgStructured.resize(initArgStructured.size());
      for (size_t i = 0; i < initArgStructured.size(); i++)
        blockArgStructured[i] = initArgStructured[i];
    }
  } else {
    llvm_unreachable("Unreachable");
  }

  if (!offsetMap.contains(operand)) {
    offsetMap[operand] = PtrOffsetInfo();
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType()))
      offsetMap[operand].setUnstructured(tensorType.getRank());
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "finish parse\n" << operand << '\n';
    auto data = offsetMap.at(operand);
    for (auto s : data.getStructuredRef())
      os << s;
    os << "\n";
  });
}

void parseArithOp(Operation *arithOp, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  assert(isa<arith::ArithDialect>(arithOp->getDialect()));
  if (auto addIOp = dyn_cast<arith::AddIOp>(arithOp)) {
    parseAddI(addIOp, loc, rewriter, offsetMap);
  } else if (auto subIOp = dyn_cast<arith::SubIOp>(arithOp)) {
    parseSubI(subIOp, loc, rewriter, offsetMap);
  } else if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(arithOp)) {
    parseIndexCast(indexCastOp, loc, rewriter, offsetMap);
  } else if (auto constantFloatOp = dyn_cast<arith::ConstantFloatOp>(arithOp)) {
    parseConstantFloat(constantFloatOp, loc, rewriter, offsetMap);
  } else if (auto constantIntOp = dyn_cast<arith::ConstantIntOp>(arithOp)) {
    parseConstantInt(constantIntOp, loc, rewriter, offsetMap);
  } else if (auto constantOp = dyn_cast<arith::ConstantOp>(arithOp)) {
    parseConstant(constantOp, loc, rewriter, offsetMap);
  } else if (auto extSIOp = dyn_cast<arith::ExtSIOp>(arithOp)) {
    parseExtSI(extSIOp, loc, rewriter, offsetMap);
  } else if (auto mulIOp = dyn_cast<arith::MulIOp>(arithOp)) {
    parseMulI(mulIOp, loc, rewriter, offsetMap);
  } else if (auto remSIOp = dyn_cast<arith::RemSIOp>(arithOp)) {
    parseRemSI(remSIOp, loc, rewriter, offsetMap);
  } else if (auto divSIOp = dyn_cast<arith::DivSIOp>(arithOp)) {
    parseDivSI(divSIOp, loc, rewriter, offsetMap);
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(arithOp)) {
    parseSelect(selectOp, loc, rewriter, offsetMap);
  } else if (auto fPToSIOp = dyn_cast<arith::FPToSIOp>(arithOp)) {
    parseFPToSI(fPToSIOp, loc, rewriter, offsetMap);
  } else if (auto sIToFPOp = dyn_cast<arith::SIToFPOp>(arithOp)) {
    parseSIToFP(sIToFPOp, loc, rewriter, offsetMap);
  } else if (auto mulFOp = dyn_cast<arith::MulFOp>(arithOp)) {
    parseMulF(mulFOp, loc, rewriter, offsetMap);
  } else if (auto divFOp = dyn_cast<arith::DivFOp>(arithOp)) {
    parseDivF(divFOp, loc, rewriter, offsetMap);
  } else if (auto addFOp = dyn_cast<arith::AddFOp>(arithOp)) {
    parseAddF(addFOp, loc, rewriter, offsetMap);
  } else if (auto subFOp = dyn_cast<arith::SubFOp>(arithOp)) {
    parseSubF(subFOp, loc, rewriter, offsetMap);
  } else if (auto minNumFOp = dyn_cast<arith::MinNumFOp>(arithOp)) {
    parseMinNumF(minNumFOp, loc, rewriter, offsetMap);
  } else if (auto maxNumFOp = dyn_cast<arith::MaxNumFOp>(arithOp)) {
    parseMaxNumF(maxNumFOp, loc, rewriter, offsetMap);
  } else if (auto maxSIOp = dyn_cast<arith::MaxSIOp>(arithOp)) {
    parseMaxSI(maxSIOp, loc, rewriter, offsetMap);
  } else if (auto minSIOp = dyn_cast<arith::MinSIOp>(arithOp)) {
    parseMinSI(minSIOp, loc, rewriter, offsetMap);
  } else if (auto cmpIOp = dyn_cast<arith::CmpIOp>(arithOp)) {
    parseCmpI(cmpIOp, loc, rewriter, offsetMap);
  } else if (auto cmpFOp = dyn_cast<arith::CmpFOp>(arithOp)) {
    parseCmpF(cmpFOp, loc, rewriter, offsetMap);
  } else if (auto andIOp = dyn_cast<arith::AndIOp>(arithOp)) {
    parseAndI(andIOp, loc, rewriter, offsetMap);
  } else if (auto orIOp = dyn_cast<arith::OrIOp>(arithOp)) {
    parseOrI(orIOp, loc, rewriter, offsetMap);
  }
}

void parseTritonOp(Operation *tritonOp, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  assert(isa<triton::TritonDialect>(tritonOp->getDialect()));
  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(tritonOp)) {
    parseAddPtr(addPtrOp, loc, rewriter, offsetMap);
  } else if (auto splatOp = dyn_cast<triton::SplatOp>(tritonOp)) {
    parseSplat(splatOp, loc, rewriter, offsetMap);
  } else if (auto getProgramIdOp = dyn_cast<triton::GetProgramIdOp>(tritonOp)) {
    parseGetProgramId(getProgramIdOp, loc, rewriter, offsetMap);
  } else if (auto getNumProgramsOp =
                 dyn_cast<triton::GetNumProgramsOp>(tritonOp)) {
    parseGetNumPrograms(getNumProgramsOp, loc, rewriter, offsetMap);
  } else if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(tritonOp)) {
    parseMakeRange(makeRangeOp, loc, rewriter, offsetMap);
  } else if (auto bitcastOp = dyn_cast<triton::BitcastOp>(tritonOp)) {
    parseBitcast(bitcastOp, loc, rewriter, offsetMap);
  } else if (auto loadOp = dyn_cast<triton::LoadOp>(tritonOp)) {
    parseLoad(loadOp, loc, rewriter, offsetMap);
  } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(tritonOp)) {
    parseBroadcast(broadcastOp, loc, rewriter, offsetMap);
  } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(tritonOp)) {
    parseExpandDims(expandDimsOp, loc, rewriter, offsetMap);
  } else if (auto clampFOp = dyn_cast<triton::ClampFOp>(tritonOp)) {
    parseClampF(clampFOp, loc, rewriter, offsetMap);
  } else if (auto makeTensorDescOp =
                 dyn_cast<triton::MakeTensorDescOp>(tritonOp)) {
    parseMakeTensorDesc(makeTensorDescOp, loc, rewriter, offsetMap);
  } else if (auto makeTensorPtrOp =
                 dyn_cast<triton::MakeTensorPtrOp>(tritonOp)) {
    parseMakeTensorPtr(makeTensorPtrOp, loc, rewriter, offsetMap);
  } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(tritonOp)) {
    parseReduce(reduceOp, loc, rewriter, offsetMap);
  } else if (auto reduceReturnOp = dyn_cast<triton::ReduceReturnOp>(tritonOp)) {
    parseReduceReturn(reduceReturnOp, loc, rewriter, offsetMap);
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(tritonOp)) {
    parseAdvance(advanceOp, loc, rewriter, offsetMap);
  } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(tritonOp)) {
    parseIntToPtr(intToPtrOp, loc, rewriter, offsetMap);
  }
}

void parseAddPtr(triton::AddPtrOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get addPtr base_ptr
  Value ptr = op.getPtr();
  parse(ptr, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo ptrOffsetInfo = offsetMap.at(ptr);
  SmallVector<bool> &ptrStructured = ptrOffsetInfo.getStructuredRef();
  // Get addPtr offset
  Value offsetValue = op.getOffset();
  parse(offsetValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo offsetOffsetInfo = offsetMap.at(offsetValue);
  SmallVector<bool> &offsetStructured = offsetOffsetInfo.getStructuredRef();
  // Modify IR

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  if (auto offsetType = dyn_cast<RankedTensorType>(offsetValue.getType())) {
    auto offsetElementType = cast<IntegerType>(offsetType.getElementType());
    if (offsetElementType.getWidth() != 64) {
      auto newOffsetType = RankedTensorType::get(offsetType.getShape(),
                                                 rewriter.getIntegerType(64));
      offsetValue = rewriter.create<arith::ExtSIOp>(op.getLoc(), newOffsetType,
                                                    offsetValue);
    }
  } else {
    auto offsetIntType = cast<IntegerType>(offsetValue.getType());
    if (offsetIntType.getWidth() != 64) {
      offsetValue = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getIntegerType(64), offsetValue);
    }
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseAddPtr] Adding offset\n";
    os << ptrOffsetInfo.getOffset() << '\n' << offsetValue << '\n';
  });
  auto offset = rewriter.create<arith::AddIOp>(
      op.getLoc(), ptrOffsetInfo.getOffset(), offsetValue);
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseAddPtr] offset is\n" << offset << '\n';
  });
  // Set addPtr offset map
  auto dst = op.getResult();
  offsetMap[dst] = {ptrOffsetInfo.getPtr(), offset};
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseAddPtr] ptrStructured: ";
    for (size_t i = 0; i < ptrStructured.size(); i++)
      os << ptrStructured[i];
    os << "\n";
    os << "[parseAddPtr] offsetStructured: ";
    for (size_t i = 0; i < offsetStructured.size(); i++)
      os << offsetStructured[i];
    os << "\n";
  });
  assert(ptrStructured.size() == offsetStructured.size() &&
         "ptrRank and offsetRank should be same");
  offsetMap[dst].setScalarLike(ptrOffsetInfo.isScalarLike() &&
                               offsetOffsetInfo.isScalarLike());
  dstStructured.resize(ptrStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = ptrStructured[i] && offsetStructured[i];
}

void parseSplat(triton::SplatOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get splat src
  auto src = op.getSrc();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  auto dst = op.getResult();
  auto dstType = cast<RankedTensorType>(dst.getType());

  offsetMap[dst] = {srcOffsetInfo.getPtr(), nullptr};

  // Modify IR
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseSplat] dst is\n" << dst << '\n';
  });
  if (isa<triton::PointerType>(dstType.getElementType())) {
    RewriterBase::InsertionGuard guard(rewriter);
    auto dstShape = dstType.getShape();
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(dstShape, rewriter.getIntegerType(64)),
        valueOffset);

    offsetMap[dst].setOffset(offset);
  }

  // Set addPtr offset map

  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getRank(), true);
  offsetMap[dst].setScalarLike(true);
}

void parseSubI(arith::SubIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get subi lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  // Get subi lhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  // Set subi offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsOffsetInfo.isScalarLike() &&
                               rhsOffsetInfo.isScalarLike());
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(lhsStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    // FIXME: offset could be non-positive
    dstStructured[i] = offsetMap[dst].isScalarLike();
}

void parseAddI(arith::AddIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get addi lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  // Get addi rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  // Set addi offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsOffsetInfo.isScalarLike() &&
                               rhsOffsetInfo.isScalarLike());
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(lhsStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = lhsStructured[i] && rhsStructured[i];
}

void parseIndexCast(arith::IndexCastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get indexCast input
  auto src = op.getIn();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set indexCast offset map
  auto dst = op.getOut();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(srcStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = srcStructured[i];
}

void parseConstantFloat(arith::ConstantFloatOp dst, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set constantFloat offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(true);
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  if (auto tensorType = dyn_cast<RankedTensorType>(dst.getResult().getType()))
    dstStructured.resize(tensorType.getRank(), true);
}

void parseConstantInt(arith::ConstantIntOp dst, const Location &loc,
                      RewriterBase &rewriter,
                      llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set constantInt offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(true);
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  if (auto tensorType = dyn_cast<RankedTensorType>(dst.getResult().getType()))
    dstStructured.resize(tensorType.getRank(), true);
}

void parseConstant(arith::ConstantOp dst, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set constant offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(true);
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  if (auto tensorType = dyn_cast<RankedTensorType>(dst.getResult().getType()))
    dstStructured.resize(tensorType.getRank(), true);
}

void parseGetProgramId(triton::GetProgramIdOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set getProgramId offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(true);
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
}

void parseGetNumPrograms(triton::GetNumProgramsOp op, const Location &loc,
                         RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set getNumPrograms offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(true);
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
}

void parseMakeRange(triton::MakeRangeOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set makeRange offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(false);
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(1);
  dstStructured[0] = true;
}

void parseExtSI(arith::ExtSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get extSI input
  auto src = op.getIn();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set extSI offset map
  auto dst = op.getOut();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(srcStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = srcStructured[i];
}

void parseBitcast(triton::BitcastOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get bitcast src
  auto src = op.getSrc();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set extSI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(srcStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = srcStructured[i];
}

void parseLoad(triton::LoadOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get load ptr
  auto ptr = op.getPtr();
  parse(ptr, op.getLoc(), rewriter, offsetMap);
  // Set load offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(offsetMap[ptr].isScalarLike());
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  auto tensorType = dyn_cast<RankedTensorType>(dst.getType());
  if (!tensorType)
    return;
  dstStructured.resize(tensorType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = false;
}

void parseMulI(arith::MulIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get muli lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get muli rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set muli offset map
  size_t maxSize = std::max(lhsStructured.size(), rhsStructured.size());
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(maxSize);
  for (size_t i = 0; i < maxSize; i++)
    if (lhsScalarLike)
      dstStructured[i] = rhsStructured[i];
    else if (rhsScalarLike)
      dstStructured[i] = lhsStructured[i];
    else
      dstStructured[i] = false;
}

void parseRemSI(arith::RemSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get AndI lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get AndI rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set remSI offset map
  auto dst = op.getResult();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  offsetMap[dst] = PtrOffsetInfo();
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = offsetMap[dst].isStructured();
}

void parseDivSI(arith::DivSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get divSI lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get divSI rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set divSI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = offsetMap[dst].isStructured();
}

void parseBroadcast(triton::BroadcastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get broadcast src
  auto src = op.getSrcMutable().get();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Get broadcast dim
  auto dst = op.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "tt.broadcast's input should be a tensor");
  auto srcType = cast<RankedTensorType>(src.getType());
  auto dstType = cast<RankedTensorType>(dst.getType());
  assert(srcType.getRank() == dstType.getRank() &&
         "rank of source shoule be equal to destnation");
  auto broadcastDim = ConverterUtils::getBroadcastDims(srcType, dstType);
  // Set broadcast offset map
  offsetMap[dst] = {/*ptr*/ srcOffsetInfo.getPtr(), /*offset*/ nullptr};
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());

  if (srcOffsetInfo.getPtr()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset = rewriter.create<triton::BroadcastOp>(
        loc,
        RankedTensorType::get(dstType.getShape(), rewriter.getIntegerType(64)),
        valueOffset);

    offsetMap[dst].setOffset(offset);
  }

  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(srcStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (llvm::find(broadcastDim, i) != broadcastDim.end())
      dstStructured[i] = true;
    else
      dstStructured[i] = srcStructured[i];
}

void parseExpandDims(triton::ExpandDimsOp op, const Location &loc,
                     RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get expandDims src
  auto src = op.getSrcMutable().get();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set expandDims offset map
  auto dst = op.getResult();
  offsetMap[dst] = {/*ptr*/ srcOffsetInfo.getPtr(), /*offset*/ nullptr};
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  if (srcOffsetInfo.getPtr()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset = rewriter.create<triton::ExpandDimsOp>(loc, valueOffset,
                                                         op.getAxisAttr());

    offsetMap[dst].setOffset(offset);
  }
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(srcStructured.size() + 1);
  size_t j = 0;
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (i == op.getAxis()) {
      dstStructured[i] = true;
    } else {
      dstStructured[i] = srcStructured[j];
      j++;
    }
}

void parseClampF(triton::ClampFOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get clampF src
  auto src = op.getX();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Get clampF min
  auto clampMin = op.getX();
  parse(clampMin, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo minOffsetInfo = offsetMap.at(clampMin);
  // Get clampF max
  auto clampMax = op.getX();
  parse(clampMax, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo maxOffsetInfo = offsetMap.at(clampMax);
  // Set clampF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike() &&
                               minOffsetInfo.isScalarLike() &&
                               maxOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseSelect(arith::SelectOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get select condition
  auto condition = op.getCondition();
  parse(condition, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo conditionOffsetInfo = offsetMap.at(condition);
  SmallVector<bool> &conditionStructured = conditionOffsetInfo.getStructuredRef();
  bool conditionScalarLike = conditionOffsetInfo.isScalarLike();
  // Get select trueValue
  auto trueValue = op.getTrueValue();
  parse(trueValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo trueValueOffsetInfo = offsetMap.at(trueValue);
  SmallVector<bool> &trueValueStructured = trueValueOffsetInfo.getStructuredRef();
  bool trueValueScalarLike = trueValueOffsetInfo.isScalarLike();
  // Get select falseValue
  auto falseValue = op.getFalseValue();
  parse(falseValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo falseValueOffsetInfo = offsetMap.at(falseValue);
  SmallVector<bool> &falseValueStructured = falseValueOffsetInfo.getStructuredRef();
  bool falseValueScalarLike = falseValueOffsetInfo.isScalarLike();
  // Set select offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(conditionScalarLike && trueValueScalarLike &&
                               falseValueScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseFPToSI(arith::FPToSIOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get FPToSI src
  auto src = op.getIn();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Set FPToSI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseSIToFP(arith::SIToFPOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get SIToFP src
  auto src = op.getIn();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Set SIToFP offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseMulF(arith::MulFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get MulF lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get MulF rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set MulF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseDivF(arith::DivFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get DivF lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get DivF rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set DivF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseAddF(arith::AddFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get AddF lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get AddF rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set AddF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseSubF(arith::SubFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get SubF lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get SubF rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set SubF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseMinNumF(arith::MinNumFOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get MinNumF lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get MinNumF rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set MinNumF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseMaxNumF(arith::MaxNumFOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get MinNumF lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get MinNumF rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set MaxNumF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseMaxSI(arith::MaxSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get MinNumF lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get MinNumF rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set MaxSI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = offsetMap[dst].isStructured();
}

void parseMinSI(arith::MinSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get MinNumF lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get MinNumF rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set MinSI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = offsetMap[dst].isStructured();
}

void parseCmpI(arith::CmpIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set CmpI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(false);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseCmpF(arith::CmpFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set CmpF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(false);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseAndI(arith::AndIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get AndI lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get AndI rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set AndI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = offsetMap[dst].isStructured();
}

void parseOrI(arith::OrIOp op, const Location &loc, RewriterBase &rewriter,
              llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get OrI lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get OrI rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set OrI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = offsetMap[dst].isStructured();
}

void parseFloor(math::FloorOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get Floor src
  auto src = op.getOperand();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Set Floor offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseCeil(math::CeilOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get Ceil src
  auto src = op.getOperand();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Set Ceil offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseMakeTensorDesc(triton::MakeTensorDescOp op, const Location &loc,
                         RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set MakeTensorDesc offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = true;
}

void parseMakeTensorPtr(triton::MakeTensorPtrOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set MakeTensorPtr offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = true;
}

void parseAdvance(triton::AdvanceOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set Advance offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = true;
}

void parseReduce(triton::ReduceOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get reduce src
  Value src = op->getOperand(0);
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set reduce offset map
  Value dst = op->getResult(0);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(dstShape.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (dstShape[i] == 1)
      dstStructured[i] = true;
    else
      dstStructured[i] = srcStructured[i];
}

void parseReduceReturn(triton::ReduceReturnOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get reduce src
  Value src = op->getOperand(0);
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set reduce offset map
  Value dst = op->getResult(0);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(dstShape.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (dstShape[i] == 1)
      dstStructured[i] = true;
    else
      dstStructured[i] = srcStructured[i];
}

void parseIf(scf::IfOp op, const Location &loc, RewriterBase &rewriter,
             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst) {
  const unsigned int index = cast<OpResult>(dst).getResultNumber();
  // Get if then region
  Block &thenBlock = op.getThenRegion().front();
  Value thenYieldedValue = thenBlock.getTerminator()->getOperand(index);
  parse(thenYieldedValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo thenOffsetInfo = offsetMap.at(thenYieldedValue);
  SmallVector<bool> &thenStructured = thenOffsetInfo.getStructuredRef();
  // Get if else region
  bool dstIsScalar = thenOffsetInfo.isScalarLike();
  SmallVector<bool> elseStructured = {};
  if (op.elseBlock()) {
    Block &elseBlock = op.getElseRegion().front();
    Value elseYieldedValue = elseBlock.getTerminator()->getOperand(index);
    parse(elseYieldedValue, op.getLoc(), rewriter, offsetMap);
    PtrOffsetInfo elseOffsetInfo = offsetMap.at(elseYieldedValue);
    elseStructured = elseOffsetInfo.getStructuredRef();
    dstIsScalar = dstIsScalar && elseOffsetInfo.isScalarLike();
  }
  // Set if offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(dstIsScalar);
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(thenStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (op.elseBlock())
      dstStructured[i] = thenStructured[i] && elseStructured[i];
    else
      dstStructured[i] = thenStructured[i];
}

void parseYield(scf::YieldOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get yield src
  for (auto src : op->getOperands())
    parse(src, op.getLoc(), rewriter, offsetMap);
}

void parseFor(scf::ForOp op, const Location &loc, RewriterBase &rewriter,
              llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst) {
  const unsigned int index = cast<OpResult>(dst).getResultNumber();
  // Get for region
  Value yieldedValue = op.getBody()->getTerminator()->getOperand(index);
  parse(yieldedValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo yieldOffsetInfo = offsetMap.at(yieldedValue);
  Value initArg = op.getInitArgs()[index];
  parse(initArg, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo initOffsetInfo = offsetMap.at(initArg);
  SmallVector<bool> &yieldStructured = yieldOffsetInfo.getStructuredRef();
  SmallVector<bool> &initStructured = initOffsetInfo.getStructuredRef();
  // Set for offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(yieldOffsetInfo.isScalarLike() &&
                               initOffsetInfo.isScalarLike());
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(yieldStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = yieldStructured[i] && initStructured[i];
}

void parseWhile(scf::WhileOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst) {
  const unsigned int index = cast<OpResult>(dst).getResultNumber();
  // Get for region
  Value yieldedValue = op.getAfterBody()->getTerminator()->getOperand(index);
  parse(yieldedValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo yieldOffsetInfo = offsetMap.at(yieldedValue);
  Value initArg = op.getInits()[index];
  parse(initArg, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo initOffsetInfo = offsetMap.at(initArg);
  SmallVector<bool> &yieldStructured = yieldOffsetInfo.getStructuredRef();
  SmallVector<bool> &initStructured = initOffsetInfo.getStructuredRef();
  // Set for offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(yieldOffsetInfo.isScalarLike() &&
                               initOffsetInfo.isScalarLike());
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(yieldStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = yieldStructured[i] && initStructured[i];
}

void parseExtractSlice(tensor::ExtractSliceOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get extractSlice src
  auto src = op.getOperand(0);
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set extractSlice offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(dstType.getShape().size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = srcStructured[i];
}

void parseExtract(tensor::ExtractOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto parentValue = op.getTensor();
  parse(parentValue, op.getLoc(), rewriter, offsetMap);
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  if (isa<triton::PointerType>(dst.getType())) {
    offsetMap[dst].setPtr(dst);
  }
  offsetMap[dst].setScalarLike(true);
}

void parseIntToPtr(triton::IntToPtrOp op, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo(dst);
  offsetMap[dst].setScalarLike(true);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  offsetMap[dst].setOffset(
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0)));
}

} // namespace triton
} // namespace mlir