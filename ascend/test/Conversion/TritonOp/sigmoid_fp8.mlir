// RUN: triton-adapter-opt --triton-linearize '--discrete-mask-access-conversion=compile-on-910-95=True force-simt-template=False' --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=True force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=True' --split-input-file %s | FileCheck %s

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<0.000000e+00> : tensor<2x29x4xf32>
		%cst_0 = arith.constant dense<1.000000e+00> : tensor<2x29x4xf32>
		%cst_1 = arith.constant dense<4> : tensor<1x29x1xi32>
		%cst_2 = arith.constant dense<4> : tensor<2x1x1xi32>
		%cst_3 = arith.constant dense<29> : tensor<2x1x1xi32>
		%c4_i32 = arith.constant 4 : i32
		%c29_i32 = arith.constant 29 : i32
		%c2_i32 = arith.constant 2 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.muli %0, %c2_i32 : i32
		%2 = tt.get_program_id y : i32
		%3 = arith.muli %2, %c29_i32 : i32
		%4 = tt.get_program_id z : i32
		%5 = arith.muli %4, %c4_i32 : i32
		%6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
		%7 = tt.splat %1 : i32 -> tensor<2xi32>
		%8 = arith.addi %6, %7 : tensor<2xi32>
		%9 = tt.make_range {end = 29 : i32, start = 0 : i32} : tensor<29xi32>
		%10 = tt.splat %3 : i32 -> tensor<29xi32>
		%11 = arith.addi %9, %10 : tensor<29xi32>
		%12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%13 = tt.splat %5 : i32 -> tensor<4xi32>
		%14 = arith.addi %12, %13 : tensor<4xi32>
		%15 = tt.expand_dims %8 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
		%16 = tt.expand_dims %15 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
		%17 = arith.muli %16, %cst_3 : tensor<2x1x1xi32>
		%18 = arith.muli %17, %cst_2 : tensor<2x1x1xi32>
		%19 = tt.expand_dims %11 {axis = 0 : i32} : tensor<29xi32> -> tensor<1x29xi32>
		%20 = tt.expand_dims %19 {axis = 2 : i32} : tensor<1x29xi32> -> tensor<1x29x1xi32>
		%21 = arith.muli %20, %cst_1 : tensor<1x29x1xi32>
		%22 = tt.broadcast %18 : tensor<2x1x1xi32> -> tensor<2x29x1xi32>
		%23 = tt.broadcast %21 : tensor<1x29x1xi32> -> tensor<2x29x1xi32>
		%24 = arith.addi %22, %23 : tensor<2x29x1xi32>
		%25 = tt.expand_dims %14 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
		%26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
		%27 = tt.broadcast %24 : tensor<2x29x1xi32> -> tensor<2x29x4xi32>
		%28 = tt.broadcast %26 : tensor<1x1x4xi32> -> tensor<2x29x4xi32>
		%29 = arith.addi %27, %28 : tensor<2x29x4xi32>
		%30 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<2x29x4x!tt.ptr<f8E4M3FN>>
		%31 = tt.addptr %30, %29 : tensor<2x29x4x!tt.ptr<f8E4M3FN>>, tensor<2x29x4xi32>
		%32 = tt.load %31 : tensor<2x29x4x!tt.ptr<f8E4M3FN>>
		%33 = tt.fp_to_fp %32 : tensor<2x29x4xf8E4M3FN> -> tensor<2x29x4xf32>
		%34 = arith.subf %cst, %33 : tensor<2x29x4xf32>
		%35 = math.exp %34 : tensor<2x29x4xf32>
		%36 = arith.addf %35, %cst_0 : tensor<2x29x4xf32>
		%37 = arith.divf %cst_0, %36 : tensor<2x29x4xf32>
		%38 = tt.fp_to_fp %37, rounding = rtne : tensor<2x29x4xf32> -> tensor<2x29x4xf8E4M3FN>
		%39 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<2x29x4x!tt.ptr<f8E4M3FN>>
		%40 = tt.addptr %39, %29 : tensor<2x29x4x!tt.ptr<f8E4M3FN>>, tensor<2x29x4xi32>
		tt.store %40, %38 : tensor<2x29x4x!tt.ptr<f8E4M3FN>>
		tt.return
	}
}

// CHECK: %[[EMPTY_INIT:[A-Za-z0-9_]+]] = tensor.empty() : tensor<2x29x4xf32>
// CHECK: %[[FILL_ZERO:[A-Za-z0-9_]+]] = linalg.fill ins(%cst_0 : f32) outs(%[[EMPTY_INIT]] : tensor<2x29x4xf32>) -> tensor<2x29x4xf32>
// CHECK: %[[FILL_ONE:[A-Za-z0-9_]+]] = linalg.fill ins(%cst : f32) outs(%[[EMPTY_INIT]] : tensor<2x29x4xf32>) -> tensor<2x29x4xf32>

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<2x29x4xf8E4M3FN>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<2x29x4xf8E4M3FN>

// CHECK: %[[FP_TO_FP:[A-Za-z0-9_]+]] = arith.extf %[[TENSOR_INPUT]] {round_mode = #hfusion.round_mode<rint>} : tensor<2x29x4xf8E4M3FN> to tensor<2x29x4xf32>
// CHECK: %[[SUB_F:[A-Za-z0-9_]+]] = arith.subf %[[FILL_ZERO]], %[[FP_TO_FP]] : tensor<2x29x4xf32>
// CHECK: %[[EXP:[A-Za-z0-9_]+]] = math.exp %[[SUB_F]] : tensor<2x29x4xf32>
// CHECK: %[[ADD_F:[A-Za-z0-9_]+]] = arith.addf %[[EXP]], %[[FILL_ONE]] : tensor<2x29x4xf32>
// CHECK: %[[DIV_F:[A-Za-z0-9_]+]] = arith.divf %[[FILL_ONE]], %[[ADD_F]] : tensor<2x29x4xf32>
// CHECK: %[[CAST_F32_TO_F8:[A-Za-z0-9_]+]] = arith.truncf %[[DIV_F]] {round_mode = #hfusion.round_mode<rint>} : tensor<2x29x4xf32> to tensor<2x29x4xf8E4M3FN>

// -----

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<0.000000e+00> : tensor<2x29x4xf32>
		%cst_0 = arith.constant dense<1.000000e+00> : tensor<2x29x4xf32>
		%cst_1 = arith.constant dense<4> : tensor<1x29x1xi32>
		%cst_2 = arith.constant dense<4> : tensor<2x1x1xi32>
		%cst_3 = arith.constant dense<29> : tensor<2x1x1xi32>
		%c4_i32 = arith.constant 4 : i32
		%c29_i32 = arith.constant 29 : i32
		%c2_i32 = arith.constant 2 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.muli %0, %c2_i32 : i32
		%2 = tt.get_program_id y : i32
		%3 = arith.muli %2, %c29_i32 : i32
		%4 = tt.get_program_id z : i32
		%5 = arith.muli %4, %c4_i32 : i32
		%6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
		%7 = tt.splat %1 : i32 -> tensor<2xi32>
		%8 = arith.addi %6, %7 : tensor<2xi32>
		%9 = tt.make_range {end = 29 : i32, start = 0 : i32} : tensor<29xi32>
		%10 = tt.splat %3 : i32 -> tensor<29xi32>
		%11 = arith.addi %9, %10 : tensor<29xi32>
		%12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%13 = tt.splat %5 : i32 -> tensor<4xi32>
		%14 = arith.addi %12, %13 : tensor<4xi32>
		%15 = tt.expand_dims %8 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
		%16 = tt.expand_dims %15 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
		%17 = arith.muli %16, %cst_3 : tensor<2x1x1xi32>
		%18 = arith.muli %17, %cst_2 : tensor<2x1x1xi32>
		%19 = tt.expand_dims %11 {axis = 0 : i32} : tensor<29xi32> -> tensor<1x29xi32>
		%20 = tt.expand_dims %19 {axis = 2 : i32} : tensor<1x29xi32> -> tensor<1x29x1xi32>
		%21 = arith.muli %20, %cst_1 : tensor<1x29x1xi32>
		%22 = tt.broadcast %18 : tensor<2x1x1xi32> -> tensor<2x29x1xi32>
		%23 = tt.broadcast %21 : tensor<1x29x1xi32> -> tensor<2x29x1xi32>
		%24 = arith.addi %22, %23 : tensor<2x29x1xi32>
		%25 = tt.expand_dims %14 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
		%26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
		%27 = tt.broadcast %24 : tensor<2x29x1xi32> -> tensor<2x29x4xi32>
		%28 = tt.broadcast %26 : tensor<1x1x4xi32> -> tensor<2x29x4xi32>
		%29 = arith.addi %27, %28 : tensor<2x29x4xi32>
		%30 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<2x29x4x!tt.ptr<f8E5M2>>
		%31 = tt.addptr %30, %29 : tensor<2x29x4x!tt.ptr<f8E5M2>>, tensor<2x29x4xi32>
		%32 = tt.load %31 : tensor<2x29x4x!tt.ptr<f8E5M2>>
		%33 = tt.fp_to_fp %32 : tensor<2x29x4xf8E5M2> -> tensor<2x29x4xf32>
		%34 = arith.subf %cst, %33 : tensor<2x29x4xf32>
		%35 = math.exp %34 : tensor<2x29x4xf32>
		%36 = arith.addf %35, %cst_0 : tensor<2x29x4xf32>
		%37 = arith.divf %cst_0, %36 : tensor<2x29x4xf32>
		%38 = tt.fp_to_fp %37, rounding = rtne : tensor<2x29x4xf32> -> tensor<2x29x4xf8E5M2>
		%39 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<2x29x4x!tt.ptr<f8E5M2>>
		%40 = tt.addptr %39, %29 : tensor<2x29x4x!tt.ptr<f8E5M2>>, tensor<2x29x4xi32>
		tt.store %40, %38 : tensor<2x29x4x!tt.ptr<f8E5M2>>
		tt.return
	}
}

// CHECK: %[[EMPTY_INIT:[A-Za-z0-9_]+]] = tensor.empty() : tensor<2x29x4xf32>
// CHECK: %[[FILL_ZERO:[A-Za-z0-9_]+]] = linalg.fill ins(%cst_0 : f32) outs(%[[EMPTY_INIT]] : tensor<2x29x4xf32>) -> tensor<2x29x4xf32>
// CHECK: %[[FILL_ONE:[A-Za-z0-9_]+]] = linalg.fill ins(%cst : f32) outs(%[[EMPTY_INIT]] : tensor<2x29x4xf32>) -> tensor<2x29x4xf32>

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<2x29x4xf8E5M2>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<2x29x4xf8E5M2>

// CHECK: %[[FP_TO_FP:[A-Za-z0-9_]+]] = arith.extf %[[TENSOR_INPUT]] {round_mode = #hfusion.round_mode<rint>} : tensor<2x29x4xf8E5M2> to tensor<2x29x4xf32>
// CHECK: %[[SUB_F:[A-Za-z0-9_]+]] = arith.subf %[[FILL_ZERO]], %[[FP_TO_FP]] : tensor<2x29x4xf32>
// CHECK: %[[EXP:[A-Za-z0-9_]+]] = math.exp %[[SUB_F]] : tensor<2x29x4xf32>
// CHECK: %[[ADD_F:[A-Za-z0-9_]+]] = arith.addf %[[EXP]], %[[FILL_ONE]] : tensor<2x29x4xf32>
// CHECK: %[[DIV_F:[A-Za-z0-9_]+]] = arith.divf %[[FILL_ONE]], %[[ADD_F]] : tensor<2x29x4xf32>
// CHECK: %[[CAST_F32_TO_F8:[A-Za-z0-9_]+]] = arith.truncf %[[DIV_F]] {round_mode = #hfusion.round_mode<rint>} : tensor<2x29x4xf32> to tensor<2x29x4xf8E5M2>