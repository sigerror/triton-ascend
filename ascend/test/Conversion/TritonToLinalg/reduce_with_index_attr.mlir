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