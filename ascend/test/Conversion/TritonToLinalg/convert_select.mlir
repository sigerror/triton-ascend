// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
// RUN: triton-adapter-opt --triton-to-linalg="named-ops=True" %s | FileCheck %s
module {
    tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) -> () {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_0 = arith.constant dense<256> : tensor<512xi64>
    %cst_1 = arith.constant dense<512> : tensor<512xi32>
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %1 = arith.cmpi slt, %0, %cst_1 : tensor<512xi32>
    %2 = arith.extsi %0 : tensor<512xi32> to tensor<512xi64>
    %3 = arith.cmpi slt, %2, %cst_0 : tensor<512xi64>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %6 = tt.load %5, %1, %cst evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %8 = tt.addptr %7, %0 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %9 = tt.load %8, %1, %cst evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>>
    %10 = arith.select %3, %6, %9 : tensor<512xi1>, tensor<512xf32>
    %11 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %12 = tt.addptr %11, %0 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    tt.store %12, %10, %1 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>>
    tt.return
    }
}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:      %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, %[[VAL_0:.*]]: memref<?xf32> {tt.tensor_kind = 0 : i32}, %[[VAL_1:.*]]: memref<?xf32> {tt.tensor_kind = 0 : i32}, %[[VAL_2:.*]]: memref<?xf32> {tt.tensor_kind = 1 : i32},
// CHECK-SAME:      %[[ARG_5:.*]]: i32, [[ARG_6:.*]]: i32, %[[ARG_7:.*]]: i32, %[[ARG_8:.*]]: i32, %[[ARG_9:.*]]: i32, %[[ARG_10:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[IN_0:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [512], strides: [1] : memref<?xf32> to memref<512xf32, strided<[1]>>
// CHECK:           %[[BUF_IN_0:.*]] = memref.alloc() : memref<512xf32>
// CHECK:           memref.copy %[[IN_0]], %[[BUF_IN_0]] : memref<512xf32, strided<[1]>> to memref<512xf32>
// CHECK:           %[[VAL_3:.*]] = bufferization.to_tensor %[[BUF_IN_0]] restrict writable : memref<512xf32>
// CHECK:           %[[IN_1:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [512], strides: [1] : memref<?xf32> to memref<512xf32, strided<[1]>>
// CHECK:           %[[BUF_IN_1:.*]] = memref.alloc() : memref<512xf32>
// CHECK:           memref.copy %[[IN_1]], %[[BUF_IN_1]] : memref<512xf32, strided<[1]>> to memref<512xf32>
// CHECK:           %[[VAL_4:.*]] = bufferization.to_tensor %[[BUF_IN_1]] restrict writable : memref<512xf32>
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %[[VAL_3]][0] [256] [1] : tensor<512xf32> to tensor<256xf32>
// CHECK:           %[[SLICE_1:.*]] = tensor.insert_slice %[[SLICE]] into %[[VAL_4]][0] [256] [1] : tensor<256xf32> into tensor<512xf32>
// CHECK:           %[[BUF_2:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [512], strides: [1] : memref<?xf32> to memref<512xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[SLICE_1]] in writable %[[BUF_2]] : (tensor<512xf32>, memref<512xf32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }