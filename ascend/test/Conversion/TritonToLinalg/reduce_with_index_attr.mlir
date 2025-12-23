// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @argmax_left_0
// CHECK: %[[REDUCED:.*]] = linalg.reduce
// CHECK: ins(%[[NEW_VALUE:.*]], %[[NEW_INDEX:.*]] : tensor<4096xf32>, tensor<4096xi32>)
// CHECK: outs(%[[OLD_VALUE:.*]], %[[OLD_INDEX:.*]] : tensor<f32>, tensor<i32>)
// CHECK: dimensions = [0]  {reduce_mode = "max_with_index", tie_break_left = "true"}
tt.func public @argmax_left_0(%arg0: !tt.ptr<i32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<4096xf32>
  %cst_0 = arith.constant dense<0> : tensor<4096xi32>
  %0:2 = "tt.reduce"(%cst, %cst_0) <{axis = 0 : i32}> ({
  ^bb0(%arg1: f32, %arg2: i32, %arg3: f32, %arg4: i32):
    %1 = arith.cmpf ogt, %arg1, %arg3 : f32
    %2 = arith.cmpf oeq, %arg1, %arg3 : f32
    %3 = arith.cmpi slt, %arg2, %arg4 : i32
    %4 = arith.andi %2, %3 : i1
    %5 = arith.ori %1, %4 : i1
    %6 = arith.select %5, %arg1, %arg3 : f32
    %7 = arith.select %5, %arg2, %arg4 : i32
    tt.reduce.return %6, %7 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  tt.store %arg0, %0#1 : !tt.ptr<i32>
  tt.return
}

// -----
// CHECK-LABEL: func.func @argmax_left_1
// CHECK: %[[REDUCED:.*]] = linalg.reduce
// CHECK: ins(%[[NEW_VALUE:.*]], %[[NEW_INDEX:.*]] : tensor<4096xf32>, tensor<4096xi32>)
// CHECK: outs(%[[OLD_VALUE:.*]], %[[OLD_INDEX:.*]] : tensor<f32>, tensor<i32>)
// CHECK: dimensions = [0]  {reduce_mode = "max_with_index", tie_break_left = "true"}
tt.func public @argmax_left_1(%arg0: !tt.ptr<i32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<4096xf32>
  %cst_0 = arith.constant dense<0> : tensor<4096xi32>
  %0:2 = "tt.reduce"(%cst, %cst_0) <{axis = 0 : i32}> ({
  ^bb0(%arg1: f32, %arg2: i32, %arg3: f32, %arg4: i32):
    %1 = arith.cmpf ogt, %arg1, %arg3 : f32
    %2 = arith.select %1, %arg1, %arg3 : f32
    %3 = arith.select %1, %arg2, %arg4 : i32
    tt.reduce.return %2, %3 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  tt.store %arg0, %0#1 : !tt.ptr<i32>
  tt.return
}

// -----
// CHECK-LABEL: func.func @argmax_right
// CHECK: %[[REDUCED:.*]] = linalg.reduce
// CHECK: ins(%[[NEW_VALUE:.*]], %[[NEW_INDEX:.*]] : tensor<4096xf32>, tensor<4096xi32>)
// CHECK: outs(%[[OLD_VALUE:.*]], %[[OLD_INDEX:.*]] : tensor<f32>, tensor<i32>)
// CHECK: dimensions = [0]  {reduce_mode = "max_with_index", tie_break_left = "false"}
tt.func public @argmax_right(%arg0: !tt.ptr<i32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<4096xf32>
  %cst_0 = arith.constant dense<0> : tensor<4096xi32>
  %0:2 = "tt.reduce"(%cst, %cst_0) <{axis = 0 : i32}> ({
  ^bb0(%arg1: f32, %arg2: i32, %arg3: f32, %arg4: i32):
    %1 = arith.cmpf ogt, %arg1, %arg3 : f32
    %2 = arith.cmpf oeq, %arg1, %arg3 : f32
    %3 = arith.cmpi sgt, %arg2, %arg4 : i32
    %4 = arith.andi %2, %3 : i1
    %5 = arith.ori %1, %4 : i1
    %6 = arith.select %5, %arg1, %arg3 : f32
    %7 = arith.select %5, %arg2, %arg4 : i32
    tt.reduce.return %6, %7 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  tt.store %arg0, %0#1 : !tt.ptr<i32>
  tt.return
}

// -----
// CHECK-LABEL: func.func @argmin_left_0
// CHECK: %[[REDUCED:.*]] = linalg.reduce
// CHECK: ins(%[[NEW_VALUE:.*]], %[[NEW_INDEX:.*]] : tensor<4096xf32>, tensor<4096xi32>)
// CHECK: outs(%[[OLD_VALUE:.*]], %[[OLD_INDEX:.*]] : tensor<f32>, tensor<i32>)
// CHECK: dimensions = [0]  {reduce_mode = "min_with_index", tie_break_left = "true"}
tt.func public @argmin_left_0(%arg0: !tt.ptr<i32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<4096xf32>
  %cst_0 = arith.constant dense<0> : tensor<4096xi32>
  %0:2 = "tt.reduce"(%cst, %cst_0) <{axis = 0 : i32}> ({
  ^bb0(%arg1: f32, %arg2: i32, %arg3: f32, %arg4: i32):
    %1 = arith.cmpf olt, %arg1, %arg3 : f32
    %2 = arith.cmpf oeq, %arg1, %arg3 : f32
    %3 = arith.cmpi slt, %arg2, %arg4 : i32
    %4 = arith.andi %2, %3 : i1
    %5 = arith.ori %1, %4 : i1
    %6 = arith.select %5, %arg1, %arg3 : f32
    %7 = arith.select %5, %arg2, %arg4 : i32
    tt.reduce.return %6, %7 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  tt.store %arg0, %0#1 : !tt.ptr<i32>
  tt.return
}

// -----
// CHECK-LABEL: func.func @argmin_left_1
// CHECK: %[[REDUCED:.*]] = linalg.reduce
// CHECK: ins(%[[NEW_VALUE:.*]], %[[NEW_INDEX:.*]] : tensor<4096xf32>, tensor<4096xi32>)
// CHECK: outs(%[[OLD_VALUE:.*]], %[[OLD_INDEX:.*]] : tensor<f32>, tensor<i32>)
// CHECK: dimensions = [0]  {reduce_mode = "min_with_index", tie_break_left = "true"}
tt.func public @argmin_left_1(%arg0: !tt.ptr<i32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<4096xf32>
  %cst_0 = arith.constant dense<0> : tensor<4096xi32>
  %0:2 = "tt.reduce"(%cst, %cst_0) <{axis = 0 : i32}> ({
  ^bb0(%arg1: f32, %arg2: i32, %arg3: f32, %arg4: i32):
    %1 = arith.cmpf olt, %arg1, %arg3 : f32
    %2 = arith.select %1, %arg1, %arg3 : f32
    %3 = arith.select %1, %arg2, %arg4 : i32
    tt.reduce.return %2, %3 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  tt.store %arg0, %0#1 : !tt.ptr<i32>
  tt.return
}

// -----
// CHECK-LABEL: func.func @argmin_right
// CHECK: %[[REDUCED:.*]] = linalg.reduce
// CHECK: ins(%[[NEW_VALUE:.*]], %[[NEW_INDEX:.*]] : tensor<4096xf32>, tensor<4096xi32>)
// CHECK: outs(%[[OLD_VALUE:.*]], %[[OLD_INDEX:.*]] : tensor<f32>, tensor<i32>)
// CHECK: dimensions = [0]  {reduce_mode = "min_with_index", tie_break_left = "false"}
tt.func public @argmin_right(%arg0: !tt.ptr<i32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<4096xf32>
  %cst_0 = arith.constant dense<0> : tensor<4096xi32>
  %0:2 = "tt.reduce"(%cst, %cst_0) <{axis = 0 : i32}> ({
  ^bb0(%arg1: f32, %arg2: i32, %arg3: f32, %arg4: i32):
    %1 = arith.cmpf olt, %arg1, %arg3 : f32
    %2 = arith.cmpf oeq, %arg1, %arg3 : f32
    %3 = arith.cmpi sgt, %arg2, %arg4 : i32
    %4 = arith.andi %2, %3 : i1
    %5 = arith.ori %1, %4 : i1
    %6 = arith.select %5, %arg1, %arg3 : f32
    %7 = arith.select %5, %arg2, %arg4 : i32
    tt.reduce.return %6, %7 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  tt.store %arg0, %0#1 : !tt.ptr<i32>
  tt.return
}

// -----
// CHECK-LABEL: func.func @reduce_shape_1_23_1_axis_1_0
// CHECK:  %[[CST_0:.*]] = arith.constant 0 : i32
// CHECK: %[[reduced:.*]]:2 = linalg.reduce ins(%[[VAL_8:.*]], %[[VAL_11:.*]] : tensor<1x23x1xf16>, tensor<1x23x1xi32>) 
// CHECK: outs(%[[VAL_12:.*]], %[[VAL_13:.*]] : tensor<1x1xf16>, tensor<1x1xi32>) dimensions = [1]  {reduce_mode = "max_with_index", tie_break_left = "true"}
// CHECK:   (%[[VAL_15:.*]]: f16, %[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: f16, %[[VAL_18:.*]]: i32) {
// CHECK:     linalg.yield %[[VAL_31:.*]], %[[VAL_32:.*]] : f16, i32
// CHECK:   }
// CHECK: %[[VAL_33:.*]] = tensor.collapse_shape %[[reduced]]#0
// CHECK: %[[VAL_34:.*]] = tensor.empty() : tensor<1xi32>
// CHECK: %[[VAL_35:.*]] = linalg.fill ins(%[[CST_0]] : i32) outs(%[[VAL_34]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK: %[[VAL_36:.*]] = memref.reinterpret_cast %[[VAL_2:.*]] to offset: [0], sizes: [1], strides: [1] : memref<?xf16> to memref<1xf16, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL_33]] in writable %[[VAL_36]] : (tensor<1xf16>, memref<1xf16, strided<[1]>>) -> ()
// CHECK: %[[VAL_37:.*]] = memref.reinterpret_cast %[[VAL_3:.*]] to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[VAL_35]] in writable %[[VAL_37]] : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
module {
  tt.func public @reduce_shape_1_23_1_axis_1_0(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true 
    %0 = tt.make_range {end = 23 : i32, start = 0 : i32} : tensor<23xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<23xi32> -> tensor<1x23xi32>
    %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<1x23xi32> -> tensor<1x23x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x23x1x!tt.ptr<f16>>
    %4 = tt.addptr %3, %2 : tensor<1x23x1x!tt.ptr<f16>>, tensor<1x23x1xi32>
    %5 = tt.load %4 : tensor<1x23x1x!tt.ptr<f16>>
    %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x23x1x!tt.ptr<i32>>
    %7 = tt.addptr %6, %2 : tensor<1x23x1x!tt.ptr<i32>>, tensor<1x23x1xi32>
    %8 = tt.load %7 : tensor<1x23x1x!tt.ptr<i32>>
    %9:2 = "tt.reduce"(%5, %8) <{axis = 1 : i32}> ({
    ^bb0(%arg4: f16, %arg5: i32, %arg6: f16, %arg7: i32):
      %15 = arith.cmpf ogt, %arg4, %arg6 : f16
      %16 = arith.cmpf oeq, %arg4, %arg6 : f16
      %17 = arith.cmpf une, %arg4, %arg4 : f16
      %18 = arith.cmpf une, %arg6, %arg6 : f16
      %19 = arith.xori %18, %true : i1 
      %20 = arith.andi %17, %19 : i1 
      %21 = arith.ori %15, %20 : i1 
      %22 = arith.andi %17, %18 : i1 
      %23 = arith.ori %16, %22 : i1 
      %24 = arith.cmpi slt, %arg5, %arg7 : i32
      %25 = arith.andi %23, %24 : i1 
      %26 = arith.ori %21, %25 : i1 
      %27 = arith.select %26, %arg4, %arg6 : f16
      %28 = arith.select %26, %arg5, %arg7 : i32
      tt.reduce.return %27, %28 : f16, i32
    }) : (tensor<1x23x1xf16>, tensor<1x23x1xi32>) -> (tensor<1x1xf16>, tensor<1x1xi32>)
    %10:2 = "tt.reduce"(%9#0, %9#1) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f16, %arg5: i32, %arg6: f16, %arg7: i32):
      %15 = arith.cmpf ogt, %arg4, %arg6 : f16
      %16 = arith.cmpf oeq, %arg4, %arg6 : f16
      %17 = arith.cmpf une, %arg4, %arg4 : f16
      %18 = arith.cmpf une, %arg6, %arg6 : f16
      %19 = arith.xori %18, %true : i1 
      %20 = arith.andi %17, %19 : i1 
      %21 = arith.ori %15, %20 : i1 
      %22 = arith.andi %17, %18 : i1 
      %23 = arith.ori %16, %22 : i1 
      %24 = arith.cmpi slt, %arg5, %arg7 : i32
      %25 = arith.andi %23, %24 : i1 
      %26 = arith.ori %21, %25 : i1 
      %27 = arith.select %26, %arg4, %arg6 : f16
      %28 = arith.select %26, %arg5, %arg7 : i32
      tt.reduce.return %27, %28 : f16, i32
    }) : (tensor<1x1xf16>, tensor<1x1xi32>) -> (tensor<1xf16>, tensor<1xi32>)
    %11 = tt.addptr %arg2, %c0_i32 : !tt.ptr<f16>, i32
    %12 = tt.splat %11 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>>
    tt.store %12, %10#0 : tensor<1x!tt.ptr<f16>>
    %13 = tt.addptr %arg3, %c0_i32 : !tt.ptr<i32>, i32
    %14 = tt.splat %13 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    tt.store %14, %10#1 : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}

