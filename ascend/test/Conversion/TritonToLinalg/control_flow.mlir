// RUN: triton-adapter-opt --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --bubble-up-operation --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @grouped_gemm_triton_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg3: i32 {tt.divisibility = 16 : i32} , %arg4: i32 {tt.divisibility = 16 : i32} , %arg5: i32 {tt.divisibility = 16 : i32} , %arg6: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg7: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg8: !tt.ptr<i64> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32 
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32> 
    %c64_i64 = arith.constant 64 : i64 
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16> 
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16> 
    %c1_i32 = arith.constant 1 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %c0_i64 = arith.constant 0 : i64 
    %cst_2 = arith.constant dense<128> : tensor<32x128xi32> 
    %cst_3 = arith.constant dense<128> : tensor<64x128xi32> 
    %c128_i32 = arith.constant 128 : i32 
    %cst_4 = arith.constant dense<2048> : tensor<32x1xi32> 
    %c3145728_i64 = arith.constant 3145728 : i64 
    %cst_5 = arith.constant dense<2048> : tensor<64x1xi64> 
    %cst_6 = arith.constant dense<0> : tensor<32xi32> 
    %cst_7 = arith.constant dense<0> : tensor<64xi32> 
    %c32_i32 = arith.constant 32 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = tt.get_program_id y : i32 
    %2 = tt.addptr %arg8, %arg3 : !tt.ptr<i64>, i32 
    %3 = tt.load %2 : !tt.ptr<i64> 
    %4 = arith.extsi %0 : i32 to i64 
    %5 = arith.cmpi sge, %4, %3 : i64 
    cf.cond_br %5, ^bb1, ^bb2 
  ^bb1:  // 2 preds: ^bb0, ^bb2
    tt.return 
  ^bb2:  // pred: ^bb0
    %6 = scf.for %arg9 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg10 = %c0_i32) -> (i32)  : i32 {
      %86 = tt.addptr %arg8, %arg9 : !tt.ptr<i64>, i32 
      %87 = tt.load %86 : !tt.ptr<i64> 
      %88 = arith.cmpi sge, %4, %87 : i64 
      %89 = arith.select %88, %arg9, %arg10 : i32 
      scf.yield %89 : i32 
    } 
    %7 = tt.addptr %arg8, %6 : !tt.ptr<i64>, i32 
    %8 = tt.load %7 : !tt.ptr<i64> 
    %9 = tt.addptr %arg6, %6 : !tt.ptr<i64>, i32 
    %10 = tt.load %9 : !tt.ptr<i64> 
    %11 = arith.subi %4, %8 : i64 
    %12 = arith.muli %11, %c64_i64 : i64 
    %13 = arith.addi %10, %12 : i64 
    %14 = tt.addptr %9, %c1_i32 : !tt.ptr<i64>, i32 
    %15 = tt.load %14 : !tt.ptr<i64> 
    %16 = arith.addi %13, %c64_i64 : i64 
    %17 = arith.minsi %15, %16 : i64 
    %18 = tt.addptr %arg7, %6 : !tt.ptr<i64>, i32 
    %19 = tt.load %18 : !tt.ptr<i64> 
    %20 = arith.subi %17, %13 : i64 
    %21 = arith.cmpi eq, %20, %c0_i64 : i64 
    cf.cond_br %21, ^bb1, ^bb3 
  ^bb3:  // pred: ^bb2
    %22 = arith.muli %1, %c32_i32 : i32 
    %23 = arith.addi %22, %c32_i32 : i32 
    %24 = arith.minsi %23, %arg4 : i32 
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
    %26 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32> 
    %27 = arith.extsi %25 : tensor<64xi32> to tensor<64xi64> 
    %28 = tt.splat %20 : i64 -> tensor<64xi64> 
    %29 = arith.cmpi slt, %27, %28 : tensor<64xi64> 
    %30 = arith.select %29, %25, %cst_7 {tt.contiguity = dense<64> : tensor<1xi32>, tt.divisibility = dense<64> : tensor<1xi32>} : tensor<64xi1>, tensor<64xi32> 
    %31 = arith.subi %24, %22 : i32 
    %32 = tt.splat %31 : i32 -> tensor<32xi32> 
    %33 = arith.cmpi slt, %26, %32 : tensor<32xi32> 
    %34 = arith.select %33, %26, %cst_6 {tt.contiguity = dense<32> : tensor<1xi32>, tt.divisibility = dense<32> : tensor<1xi32>} : tensor<32xi1>, tensor<32xi32> 
    %35 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32> 
    %36 = tt.expand_dims %30 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32> 
    %37 = arith.extsi %36 : tensor<64x1xi32> to tensor<64x1xi64> 
    %38 = tt.splat %13 : i64 -> tensor<64x1xi64> 
    %39 = arith.addi %38, %37 : tensor<64x1xi64> 
    %40 = arith.muli %39, %cst_5 : tensor<64x1xi64> 
    %41 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>> 
    %42 = tt.addptr %41, %40 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi64> 
    %43 = tt.expand_dims %35 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32> 
    %44 = tt.broadcast %42 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>> 
    %45 = tt.broadcast %43 : tensor<1x128xi32> -> tensor<64x128xi32> 
    %46 = tt.addptr %44, %45 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32> 
    %47 = arith.muli %19, %c3145728_i64 : i64 
    %48 = tt.expand_dims %34 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32> 
    %49 = tt.splat %22 : i32 -> tensor<32x1xi32> 
    %50 = arith.addi %49, %48 : tensor<32x1xi32> 
    %51 = arith.muli %50, %cst_4 : tensor<32x1xi32> 
    %52 = arith.extsi %51 : tensor<32x1xi32> to tensor<32x1xi64> 
    %53 = tt.splat %47 : i64 -> tensor<32x1xi64> 
    %54 = arith.addi %53, %52 : tensor<32x1xi64> 
    %55 = arith.extsi %43 : tensor<1x128xi32> to tensor<1x128xi64> 
    %56 = tt.broadcast %54 : tensor<32x1xi64> -> tensor<32x128xi64> 
    %57 = tt.broadcast %55 : tensor<1x128xi64> -> tensor<32x128xi64> 
    %58 = arith.addi %56, %57 : tensor<32x128xi64> 
    %59 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x128x!tt.ptr<bf16>> 
    %60 = tt.addptr %59, %58 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi64> 
    %61 = arith.addi %arg5, %c127_i32 : i32 
    %62 = arith.divsi %61, %c128_i32 : i32 
    %63:3 = scf.for %arg9 = %c0_i32 to %62 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %46, %arg12 = %60) -> (tensor<64x32xf32>, tensor<64x128x!tt.ptr<bf16>>, tensor<32x128x!tt.ptr<bf16>>)  : i32 {
      %86 = arith.muli %arg9, %c128_i32 : i32 
      %87 = arith.subi %arg5, %86 : i32 
      %88 = tt.splat %87 : i32 -> tensor<1x128xi32> 
      %89 = arith.cmpi slt, %43, %88 : tensor<1x128xi32> 
      %90 = tt.broadcast %89 : tensor<1x128xi1> -> tensor<64x128xi1> 
      %91 = tt.load %arg11, %90, %cst_1 : tensor<64x128x!tt.ptr<bf16>> 
      %92 = tt.broadcast %89 : tensor<1x128xi1> -> tensor<32x128xi1> 
      %93 = tt.load %arg12, %92, %cst_0 : tensor<32x128x!tt.ptr<bf16>> 
      %94 = tt.trans %93 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16> 
      %95 = tt.dot %91, %94, %arg10 : tensor<64x128xbf16> * tensor<128x32xbf16> -> tensor<64x32xf32> 
      %96 = tt.addptr %arg11, %cst_3 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32> 
      %97 = tt.addptr %arg12, %cst_2 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32> 
      scf.yield %95, %96, %97 : tensor<64x32xf32>, tensor<64x128x!tt.ptr<bf16>>, tensor<32x128x!tt.ptr<bf16>> 
    } 
    %64 = arith.truncf %63#0 : tensor<64x32xf32> to tensor<64x32xbf16> 
    %65 = tt.splat %13 : i64 -> tensor<64xi64> 
    %66 = arith.addi %65, %27 : tensor<64xi64> 
    %67 = tt.splat %22 : i32 -> tensor<32xi32> 
    %68 = arith.addi %67, %26 : tensor<32xi32> 
    %69 = tt.expand_dims %66 {axis = 1 : i32} : tensor<64xi64> -> tensor<64x1xi64> 
    %70 = arith.extsi %arg4 : i32 to i64 
    %71 = tt.splat %70 : i64 -> tensor<64x1xi64> 
    %72 = arith.muli %69, %71 : tensor<64x1xi64> 
    %73 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>> 
    %74 = tt.addptr %73, %72 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi64> 
    %75 = tt.expand_dims %68 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32> 
    %76 = tt.broadcast %74 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x32x!tt.ptr<bf16>> 
    %77 = tt.broadcast %75 : tensor<1x32xi32> -> tensor<64x32xi32> 
    %78 = tt.addptr %76, %77 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32> 
    %79 = tt.splat %17 : i64 -> tensor<64x1xi64> 
    %80 = arith.cmpi slt, %69, %79 : tensor<64x1xi64> 
    %81 = tt.splat %24 : i32 -> tensor<1x32xi32> 
    %82 = arith.cmpi slt, %75, %81 : tensor<1x32xi32> 
    %83 = tt.broadcast %80 : tensor<64x1xi1> -> tensor<64x32xi1> 
    %84 = tt.broadcast %82 : tensor<1x32xi1> -> tensor<64x32xi1> 
    %85 = arith.andi %83, %84 : tensor<64x32xi1> 
    tt.store %78, %64, %85 : tensor<64x32x!tt.ptr<bf16>> 
    tt.return 
  } 
}

//CHECK-LABEL: @grouped_gemm_triton_kernel
//CHECK-NOT: cf.cond_br
//CHECK: %[[COND0:.*]] = arith.cmpi sge, %[[VAL0:.*]], %[[VAL1:.*]] : i64
//CHECK: scf.if %[[COND0]] {
//CHECK: } else {
//CHECK:   %[[COND1:.*]] = arith.cmpi eq, %[[VAL2:.*]], %[[VAL3:.*]] : i64
//CHECK:   scf.if %[[COND1]] {
//CHECK:   } else {
//CHECK:   }
//CHECK: }

