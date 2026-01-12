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

# Disable deprecated-declarations warning for C++17 compatibility
target_compile_options(TritonLinearize PRIVATE -Wno-error=deprecated-declarations)
