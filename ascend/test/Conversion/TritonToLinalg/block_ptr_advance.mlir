// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @matmul_kernel_with_block_pointers_01234567891011(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : bf16
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = arith.extsi %arg7 : i32 to i64
    %4 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %3], [%arg12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    %5 = tt.advance %4, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
    %6 = tt.splat %cst : bf16 -> tensor<128x64xbf16>
    %7:3 = scf.for %arg14 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg15 = %6, %arg16 = %5, %arg17 = %4) -> (tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>)  : i32 {
      %13 = tt.load %arg16 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>>
      %14 = tt.load %arg17 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>>
      %15 = arith.addf %13, %14 : tensor<128x64xbf16>
      %16 = arith.addf %arg15, %15 : tensor<128x64xbf16>
      %17 = tt.advance %arg16, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
      %18 = tt.advance %arg17, [%c64_i32, %c0_i32] : <tensor<128x64xbf16>>
      scf.yield %16, %17, %18 : tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>
    }
    %8 = arith.extsi %arg10 : i32 to i64
    %9 = arith.extsi %arg11 : i32 to i64
    %10 = arith.extsi %arg4 : i32 to i64
    %11 = arith.muli %arg13, %c256_i32 : i32
    %12 = tt.make_tensor_ptr %arg2, [%0, %10], [%8, %9], [%arg12, %11] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    tt.store %12, %7#0 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
    tt.return
  }
}

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK:   func.func @matmul_kernel_with_block_pointers_01234567891011
// CHECK:     %c128 = arith.constant 128 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c64 = arith.constant 64 : index
// CHECK:     %c256_i32 = arith.constant 256 : i32
// CHECK:     %c0_i32 = arith.constant 0 : i32
// CHECK:     %c64_i32 = arith.constant 64 : i32
// CHECK:     %cst = arith.constant 0.000000e+00 : bf16
// CHECK:     %0 = tensor.empty() : tensor<128x64xbf16>
// CHECK:     %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
// CHECK:     %2 = arith.index_cast [[ARG_12:%.+]] : i32 to index
// CHECK:     %3 = arith.index_cast [[ARG_6:%.+]] : i32 to index
// CHECK:     %4 = arith.index_cast [[ARG_7:%.+]] : i32 to index
// CHECK:     %5 = arith.muli %2, %3 : index
// CHECK:     %6 = arith.index_cast [[ARG_3:%.+]] : i32 to index
// CHECK:     %7 = arith.index_cast [[ARG_5:%.+]] : i32 to index
// CHECK:     %8 = arith.muli %4, %c64 : index
// CHECK:     %9 = arith.addi %5, %8 : index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [%9], sizes: [128, 64], strides: [%3, %4] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [%5], sizes: [128, 64], strides: [%3, %4] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:     %10:7 = scf.for %arg22 = %c0_i32 to [[ARG_5:%.+]] step %c64_i32 iter_args(%arg23 = %1, %arg24 = %reinterpret_cast, %arg25 = %reinterpret_cast_0, %arg26 = %9, %arg27 = %c0, %arg28 = %5, %arg29 = %c0) -> (tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
// CHECK:     %alloc = memref.alloc() : memref<128x64xbf16>
// CHECK:     %18 = arith.divsi %arg26, %3 : index
// CHECK:     %19 = arith.subi %6, %18 : index
// CHECK:     %20 = arith.maxsi %19, %c0 : index
// CHECK:     %21 = arith.minsi %20, %c128 : index
// CHECK:     %22 = arith.remsi %arg26, %3 : index
// CHECK:     %23 = arith.divsi %22, %4 : index
// CHECK:     %24 = arith.subi %7, %23 : index
// CHECK:     %25 = arith.maxsi %24, %c0 : index
// CHECK:     %26 = arith.minsi %25, %c64 : index
// CHECK:       %subview = memref.subview %arg24[0, 0] [%21, %26] [1, 1] : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
// CHECK:        %subview_2 = memref.subview %alloc[0, 0] [%21, %26] [1, 1] : memref<128x64xbf16> to memref<?x?xbf16, strided<[64, 1]>>
// CHECK:         memref.copy %subview, %subview_2 : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[64, 1]>>
// CHECK: %27 = bufferization.to_tensor %alloc restrict writable : memref<128x64xbf16>
// CHECK:       %alloc_3 = memref.alloc() : memref<128x64xbf16>
// CHECK:       %28 = arith.divsi %arg28, %3 : index
// CHECK:       %29 = arith.subi %6, %28 : index
// CHECK:       %30 = arith.maxsi %29, %c0 : index
// CHECK:       %31 = arith.minsi %30, %c128 : index
// CHECK:       %32 = arith.remsi %arg28, %3 : index
// CHECK:       %33 = arith.divsi %32, %4 : index
// CHECK:       %34 = arith.subi %7, %33 : index
// CHECK:       %35 = arith.maxsi %34, %c0 : index
// CHECK:       %36 = arith.minsi %35, %c64 : index
// CHECK:       %subview_4 = memref.subview %arg25[0, 0] [%31, %36] [1, 1] : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
// CHECK:       %subview_5 = memref.subview %alloc_3[0, 0] [%31, %36] [1, 1] : memref<128x64xbf16> to memref<?x?xbf16, strided<[64, 1]>>
// CHECK:       memref.copy %subview_4, %subview_5 : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[64, 1]>>
// CHECK:      %37 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x64xbf16>
// CHECK:%38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%27, %37 : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%27 : tensor<128x64xbf16>) {
// CHECK:      ^bb0(%in: bf16, %in_8: bf16, %out: bf16):
// CHECK:       %46 = arith.addf %in, %in_8 : bf16
// CHECK:        linalg.yield %46 : bf16
// CHECK:      } -> tensor<128x64xbf16>
// CHECK:      %39 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg23, %38 : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%arg23 : tensor<128x64xbf16>) {
// CHECK:      ^bb0(%in: bf16, %in_8: bf16, %out: bf16):
// CHECK:        %46 = arith.addf %in, %in_8 : bf16
// CHECK:        linalg.yield %46 : bf16
// CHECK:      } -> tensor<128x64xbf16>
// CHECK:      %40 = arith.muli %4, %c64 : index
// CHECK:      %41 = arith.addi %40, %arg27 : index
// CHECK:      %42 = arith.addi %arg26, %41 : index
// CHECK:      %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [128, 64], strides: [%3, %4] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:      %43 = arith.muli %3, %c64 : index
// CHECK:      %44 = arith.addi %43, %arg28 : index
// CHECK:     %45 = arith.addi %44, %arg29 : index
// CHECK:      %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%45], sizes: [128, 64], strides: [%3, %4] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:      scf.yield %39, %reinterpret_cast_6, %reinterpret_cast_7, %42, %c0, %45, %c0 : tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, index, index, index, index
// CHECK:    }
// CHECK:    %11 = arith.muli %arg15, %c256_i32 : i32
// CHECK:    %12 = arith.index_cast %11 : i32 to index
// CHECK:    %13 = arith.index_cast %arg12 : i32 to index
// CHECK:   %14 = arith.index_cast %arg13 : i32 to index
// CHECK:    %15 = arith.muli %2, %13 : index
// CHECK:    %16 = arith.muli %12, %14 : index
// CHECK:    %17 = arith.addi %15, %16 : index
// CHECK:   %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [128, 64], strides: [%13, %14] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:    bufferization.materialize_in_destination %10#0 in writable %reinterpret_cast_1 : (tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>) -> ()
// CHECK:     return
// CHECK:   }


