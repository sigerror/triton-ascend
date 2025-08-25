#ifndef TRITON_ANALYSIS_OFFSETANALYSIS_H
#define TRITON_ANALYSIS_OFFSETANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {

struct PtrOffsetInfo {
  /**
  Possible status of the ptr offset:
   - ScalarLike:
      - Tensor's elements are all the same such as [[2.0,2.0,2.0],[2.0,2.0,2.0]]
      - Constant integer or floating-point such as 2, 2.0, and `load
  tensor<1xptr>`
   - Unstructured:
      - Not a `ScalarLike` ptr offset
      - Or satisfy any below conditions:
        - Incontinuous stride such as
          - `muli [0,1,2,3] [0,1,2,3]` => [0,1,4,9]
          - `divsi [9,8,7] [3,2,1]` => [3,4,7]
          - `minsi [3,4,5] [5,4,3]` => [3,4,3]
        - From non-`scalarLike` floating point element type such as
          - `fptosi [1.0,2.0,3.0]` => [1,2,3]
        - Compilation time unknown value
          - `load %ptr, %offset` => %value
    - Structured:
      - orthongonal to `Unstructured`
        - if PtrOffsetInfo isn't `Unstructured`, it is `Structured`

  In short:
  ScalarLike ⊆ Structured
  Unstructured = {x| x ∉ Structured}

  Example:
  ```
  %y = sitofp %x
  %z = fptosi %y
  ```
  If %x is scalarLike (structured), %z will be scalar (structured) as well.
  If %x is non-scalarLike structured, %z will be unstructured.
  */

public:
  PtrOffsetInfo(Value ptr = nullptr, Value offset = nullptr);

  Value getPtr() const;
  Value getOffset() const;
  bool isScalarLike() const;
  SmallVector<bool> &getStructuredRef();
  const SmallVector<bool> &getStructured() const;

  void setPtr(const Value &ptr);
  void setOffset(const Value &offset);
  void setUnstructured(int rank);
  void setStructured(int rank);
  void setScalarLike(bool scalarLike);

  bool isStructured(int dim) const;
  bool isStructured() const;
  bool isUnstructured() const;

private:
  Value ptr;
  Value offset;

  bool scalarLike;

  SmallVector<bool> structured;
};

void parse(Value operand, const Location &loc, RewriterBase &rewriter,
           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseArithOp(Operation *arithOp, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseTritonOp(Operation *tritonOp, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseTritonOp(Operation *tritonOp, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseAddPtr(triton::AddPtrOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseSplat(triton::SplatOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseSubI(arith::SubIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseAddI(arith::AddIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseIndexCast(arith::IndexCastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseConstantFloat(arith::ConstantFloatOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseConstantInt(arith::ConstantIntOp op, const Location &loc,
                      RewriterBase &rewriter,
                      llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseConstant(arith::ConstantOp op, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseGetProgramId(triton::GetProgramIdOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseGetNumPrograms(triton::GetNumProgramsOp op, const Location &loc,
                         RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMakeRange(triton::MakeRangeOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseExtSI(arith::ExtSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseBitcast(triton::BitcastOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseLoad(triton::LoadOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMulI(arith::MulIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseRemSI(arith::RemSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseDivSI(arith::DivSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseBroadcast(triton::BroadcastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseExpandDims(triton::ExpandDimsOp op, const Location &loc,
                     RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseClampF(triton::ClampFOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseSelect(arith::SelectOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseFPToSI(arith::FPToSIOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseSIToFP(arith::SIToFPOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMulF(arith::MulFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseDivF(arith::DivFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseAddF(arith::AddFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseSubF(arith::SubFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMinNumF(arith::MinNumFOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMaxNumF(arith::MaxNumFOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMaxSI(arith::MaxSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMinSI(arith::MinSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseCmpI(arith::CmpIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseCmpF(arith::CmpFOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseAndI(arith::AndIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseOrI(arith::OrIOp op, const Location &loc, RewriterBase &rewriter,
              llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseFloor(math::FloorOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseCeil(math::CeilOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMakeTensorDesc(triton::MakeTensorDescOp op, const Location &loc,
                         RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMakeTensorPtr(triton::MakeTensorPtrOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseAdvance(triton::AdvanceOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseReduce(triton::ReduceOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseReduceReturn(triton::ReduceReturnOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseIf(scf::IfOp op, const Location &loc, RewriterBase &rewriter,
             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst);

void parseYield(scf::YieldOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseFor(scf::ForOp op, const Location &loc, RewriterBase &rewriter,
              llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst);

void parseWhile(scf::WhileOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst);

void parseExtractSlice(tensor::ExtractSliceOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseExtract(tensor::ExtractOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseIntToPtr(triton::IntToPtrOp op, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);
} // namespace triton

} // namespace mlir

#endif