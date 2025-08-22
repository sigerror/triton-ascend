// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
tt.func public @test_divsi(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) {
    %c4_i32 = arith.constant dense<4> : tensor<64xi32>
    %c256_i32 = arith.constant 256 : i32
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<64x!tt.ptr<i64>>
    %4 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<64x!tt.ptr<i64>>
    %5 = tt.splat %1 : i32 -> tensor<64xi32>
    %6 = arith.muli %arg2, %c256_i32 : i32
    %7 = tt.splat %6 : i32 -> tensor<64xi32>
    %8 = arith.divsi %7, %c4_i32 : tensor<64xi32>
    %9 = arith.addi %5, %8 : tensor<64xi32>
    %10 = arith.addi %9, %2 : tensor<64xi32>
    // CHECK: %[[SRC:.*]] = memref.reinterpret_cast
    %11 = tt.addptr %3, %10 : tensor<64x!tt.ptr<i64>>, tensor<64xi32>
    %12 = tt.load %11 : tensor<64x!tt.ptr<i64>>
    %13 = tt.addptr %4, %10 : tensor<64x!tt.ptr<i64>>, tensor<64xi32>
    tt.store %13, %12 : tensor<64x!tt.ptr<i64>>
    tt.return
}

