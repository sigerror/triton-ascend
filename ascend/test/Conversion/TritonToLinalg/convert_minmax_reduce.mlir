// RUN: triton-adapter-opt --triton-to-linalg --split-input-file %s | FileCheck %s
module {
  tt.func public @minmax_sgt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi sgt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_sgt(%[[VAL_0:.*]]: memref<?xi32>
// CHECK:    %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:    %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:    %[[VAL_3:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_2]] : i32) outs(%[[VAL_3]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_5:.*]] = tensor.empty() : tensor<i32>
// CHECK:    %[[VAL_6:.*]] = linalg.reduce ins(%[[VAL_4]] : tensor<4096xi32>) outs(%[[VAL_5]] : tensor<i32>) dimensions = [0]
// CHECK:      (%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32) {
// CHECK:        %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:        %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:        linalg.yield %[[VAL_10]] : i32
// CHECK:      }
// CHECK:    %[[VAL_11:.*]] = tensor.extract %[[VAL_6]][] : tensor<i32>
// CHECK:    %[[VAL_12:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:    %[[VAL_13:.*]] = tensor.insert %[[VAL_11]] into %[[VAL_12]]{{\[}}%[[VAL_1]]] : tensor<1xi32>
// CHECK:    %[[VAL_14:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_13]] in writable %[[VAL_14]] : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:    return


// -----

module {
  tt.func public @minmax_ugt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi ugt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_ugt(%[[VAL_0:.*]]: memref<?xi32>
// CHECK:    %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:    %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:    %[[VAL_3:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_2]] : i32) outs(%[[VAL_3]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_5:.*]] = tensor.empty() : tensor<i32>
// CHECK:    %[[VAL_6:.*]] = linalg.reduce ins(%[[VAL_4]] : tensor<4096xi32>) outs(%[[VAL_5]] : tensor<i32>) dimensions = [0]
// CHECK:      (%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32) {
// CHECK:        %[[VAL_9:.*]] = arith.cmpi ugt, %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:        %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:        linalg.yield %[[VAL_10]] : i32
// CHECK:      }
// CHECK:    %[[VAL_11:.*]] = tensor.extract %[[VAL_6]][] : tensor<i32>
// CHECK:    %[[VAL_12:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:    %[[VAL_13:.*]] = tensor.insert %[[VAL_11]] into %[[VAL_12]]{{\[}}%[[VAL_1]]] : tensor<1xi32>
// CHECK:    %[[VAL_14:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_13]] in writable %[[VAL_14]] : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:    return

// -----

module {
  tt.func public @minmax_slt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi slt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_slt(%[[VAL_0:.*]]: memref<?xi32>
// CHECK:    %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:    %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:    %[[VAL_3:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_2]] : i32) outs(%[[VAL_3]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_5:.*]] = tensor.empty() : tensor<i32>
// CHECK:    %[[VAL_6:.*]] = linalg.reduce ins(%[[VAL_4]] : tensor<4096xi32>) outs(%[[VAL_5]] : tensor<i32>) dimensions = [0]
// CHECK:      (%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32) {
// CHECK:        %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:        %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:        linalg.yield %[[VAL_10]] : i32
// CHECK:      }
// CHECK:    %[[VAL_11:.*]] = tensor.extract %[[VAL_6]][] : tensor<i32>
// CHECK:    %[[VAL_12:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:    %[[VAL_13:.*]] = tensor.insert %[[VAL_11]] into %[[VAL_12]]{{\[}}%[[VAL_1]]] : tensor<1xi32>
// CHECK:    %[[VAL_14:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_13]] in writable %[[VAL_14]] : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:    return

// -----

module {
  tt.func public @minmax_ult(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi ult, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_ult(%[[VAL_0:.*]]: memref<?xi32>
// CHECK:    %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:    %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:    %[[VAL_3:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_2]] : i32) outs(%[[VAL_3]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_5:.*]] = tensor.empty() : tensor<i32>
// CHECK:    %[[VAL_6:.*]] = linalg.reduce ins(%[[VAL_4]] : tensor<4096xi32>) outs(%[[VAL_5]] : tensor<i32>) dimensions = [0]
// CHECK:      (%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32) {
// CHECK:        %[[VAL_9:.*]] = arith.cmpi ult, %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:        %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:        linalg.yield %[[VAL_10]] : i32
// CHECK:      }
// CHECK:    %[[VAL_11:.*]] = tensor.extract %[[VAL_6]][] : tensor<i32>
// CHECK:    %[[VAL_12:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:    %[[VAL_13:.*]] = tensor.insert %[[VAL_11]] into %[[VAL_12]]{{\[}}%[[VAL_1]]] : tensor<1xi32>
// CHECK:    %[[VAL_14:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_13]] in writable %[[VAL_14]] : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:    return
