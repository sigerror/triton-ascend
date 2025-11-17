// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s

module {
  tt.func public @fn_npu_trans(%arg0: !tt.ptr<i64> , %arg1: !tt.ptr<i64> ) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32> 
    %cst_0 = arith.constant dense<4> : tensor<8x1x1xi32> 
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32> 
    %4 = arith.muli %3, %cst_1 : tensor<8x1x1xi32> 
    %5 = arith.muli %4, %cst_0 : tensor<8x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %8 = arith.muli %7, %cst : tensor<1x8x1xi32> 
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32> 
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32> 
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32> 
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32> 
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<8x8x4x!tt.ptr<i64>> 
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<i64>>, tensor<8x8x4xi32> 
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<i64>> 
    %20 = tt.trans %19 {order = array<i32: 1, 0, 2>} : tensor<8x8x4xi64> -> tensor<8x8x4xi64>
    %21 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x8x4x!tt.ptr<i64>>
    %22 = tt.addptr %21, %16 : tensor<8x8x4x!tt.ptr<i64>>, tensor<8x8x4xi32> 
    tt.store %22, %20 : tensor<8x8x4x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_npu_trans
// CHECK-SAME: %arg0: memref<?xi64>
// CHECK-SAME: %arg1: memref<?xi64>

// CHECK: %alloc = memref.alloc() : memref<8x8x4xi64> 
// CHECK: %1 = tensor.empty() : tensor<8x8x4xi64>
// CHECK: bufferization.materialize_in_destination %transposed in writable %reinterpret_cast_0 : (tensor<8x8x4xi64>, memref<8x8x4xi64, strided<[32, 4, 1]>>) -> () 

// -----

module {
  tt.func public @fn_npu_trans(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i16>)  {
    %cst = arith.constant dense<3> : tensor<1x2x1xi32> 
    %cst_0 = arith.constant dense<3> : tensor<2x1x1xi32> 
    %cst_1 = arith.constant dense<2> : tensor<2x1x1xi32> 
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> 
    %1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32> 
    %4 = arith.muli %3, %cst_1 : tensor<2x1x1xi32> 
    %5 = arith.muli %4, %cst_0 : tensor<2x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
    %8 = arith.muli %7, %cst : tensor<1x2x1xi32> 
    %9 = tt.broadcast %5 : tensor<2x1x1xi32> -> tensor<2x2x1xi32>
    %10 = tt.broadcast %8 : tensor<1x2x1xi32> -> tensor<2x2x1xi32> 
    %11 = arith.addi %9, %10 : tensor<2x2x1xi32> 
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x3xi32> -> tensor<1x1x3xi32>
    %14 = tt.broadcast %11 : tensor<2x2x1xi32> -> tensor<2x2x3xi32>
    %15 = tt.broadcast %13 : tensor<1x1x3xi32> -> tensor<2x2x3xi32>
    %16 = arith.addi %14, %15 : tensor<2x2x3xi32>
    %17 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<2x2x3x!tt.ptr<i16>>
    %18 = tt.addptr %17, %16 : tensor<2x2x3x!tt.ptr<i16>>, tensor<2x2x3xi32> 
    %19 = tt.load %18 : tensor<2x2x3x!tt.ptr<i16>>
    %20 = tt.trans %19 {order = array<i32: 1, 0, 2>} : tensor<2x2x3xi16> -> tensor<2x2x3xi16>
    %21 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<2x2x3x!tt.ptr<i16>>
    %22 = tt.addptr %21, %16 : tensor<2x2x3x!tt.ptr<i16>>, tensor<2x2x3xi32>
    tt.store %22, %20 : tensor<2x2x3x!tt.ptr<i16>>
    tt.return
  } 
} 
// CHECK-LABEL:   func.func @fn_npu_trans
// CHECK-SAME: %arg0: memref<?xi16>
// CHECK-SAME: %arg1: memref<?xi16>
// CHECK: bufferization.materialize_in_destination %transposed in writable %reinterpret_cast_0 : (tensor<2x2x3xi16>, memref<2x2x3xi16, strided<[6, 3, 1]>>) -> () 


module {
  tt.func public @fn_npu_trans(%arg0: !tt.ptr<i32> , %arg1: !tt.ptr<i32> ) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32> 
    %cst_0 = arith.constant dense<4> : tensor<8x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32> 
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> 
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> 
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32> 
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32> 
    %4 = arith.muli %3, %cst_1 : tensor<8x1x1xi32>
    %5 = arith.muli %4, %cst_0 : tensor<8x1x1xi32> 
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32> 
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32> 
    %8 = arith.muli %7, %cst : tensor<1x8x1xi32> 
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32> 
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32> 
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32> 
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32> 
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32> 
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32> 
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32> 
    %17 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x8x4x!tt.ptr<i32>> 
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<i32>>, tensor<8x8x4xi32> 
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<i32>> 
    %20 = tt.trans %19 {order = array<i32: 1, 0, 2>} : tensor<8x8x4xi32> -> tensor<8x8x4xi32>
    %21 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x8x4x!tt.ptr<i32>> 
    %22 = tt.addptr %21, %16 : tensor<8x8x4x!tt.ptr<i32>>, tensor<8x8x4xi32> 
    tt.store %22, %20 : tensor<8x8x4x!tt.ptr<i32>> 
    tt.return 
  }
} 

// CHECK-LABEL:   func.func @fn_npu_trans
// CHECK-SAME: %arg0: memref<?xi32>
// CHECK-SAME: %arg1: memref<?xi32>
// bufferization.materialize_in_destination %transposed in writable %reinterpret_cast_0 : (tensor<8x8x4xi32>, memref<8x8x4xi32, strided<[32, 4, 1]>>) -> ()
