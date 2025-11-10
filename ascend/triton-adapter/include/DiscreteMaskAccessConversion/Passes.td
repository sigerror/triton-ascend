/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef DISCRETE_MASK_ACCESS_CONVERSION_PASSES
#define DISCRETE_MASK_ACCESS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def DiscreteMaskAccessConversion : Pass<"discrete-mask-access-conversion", "mlir::ModuleOp"> {
  let summary = "Recognize and convert discrete mask memory access";
  let constructor = "triton::createDiscreteMaskAccessConversionPass()";
  let options = [
    Option<"compileOnA5", "compile-on-a5", 
        "bool", /*default*/"false",
        "compile on a5">,
    Option<"forceSimtTemplate", "force_simt_template", 
        "bool", /*default*/"false",
        "force to use simt template">
  ];
}

#endif // DISCRETE_MASK_ACCESS_CONVERSION_PASSES
