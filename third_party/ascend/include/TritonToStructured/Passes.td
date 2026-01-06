#ifndef TRITON_TO_STRUCTURED_PASSES
#define TRITON_TO_STRUCTURED_PASSES

include "mlir/Pass/PassBase.td"

def TritonToStructured : Pass<"triton-to-structured", "mlir::ModuleOp"> {
    let summary = "remove reminder/divider and reproduce addptr/mask expression ";
    let constructor = "triton::createTritonToStructuredPass()";
    let options = [
        Option<"reserved", "reserved", 
            "bool", /*default*/"true",
            "reserved option">,

    ];
}

#endif // TRITON_TO_STRUCTURED_PASSES