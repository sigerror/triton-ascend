// RUN: triton-adapter-opt --triton-to-unstructure --bubble-up-operation --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @parallel_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false}{
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %0 = tt.addptr %arg0, %arg2 : !tt.ptr<i32>, i32
      %1 = tt.load %0 : !tt.ptr<i32>
      %2 = tt.addptr %arg1, %arg2 : !tt.ptr<i32>, i32
      tt.store %2, %1 : !tt.ptr<i32>
    // CHECK: {hivm.parallel_loop}
    } {hivm.parallel_loop}
    tt.return
  }
}