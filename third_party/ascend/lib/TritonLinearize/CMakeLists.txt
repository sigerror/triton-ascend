add_triton_library(TritonLinearize
  TritonLinearize.cpp
  TritonLinearizePass.cpp
  PtrAnalysis.cpp
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp


  DEPENDS
  TritonStructuredTableGen
  TritonLinearizePassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRReconcileUnrealizedCasts
  TritonIR
  TritonTransforms
  TritonStructuredIR
)
