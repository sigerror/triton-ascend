// RUN: triton-adapter-opt --triton-to-unstructure %s | FileCheck %s

tt.func public @test_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %c3_i32 = arith.constant 3 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<128> : tensor<128xi32>
  %cst_0 = arith.constant dense<0> : tensor<128xi32>
  %cst_1 = arith.constant dense<300> : tensor<128xi32>
  %c128_i32 = arith.constant 128 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c128_i32 : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
  %4 = arith.addi %3, %2 : tensor<128xi32>
  %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %7 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %9 = tt.load %8 : tensor<128x!tt.ptr<i32>>
  %10 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %11 = tt.addptr %10, %9 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %12:3 = scf.for %arg4 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg5 = %4, %arg6 = %6, %arg7 = %11) -> (tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>)  : i32 {
    %13:3 = scf.for %arg8 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg9 = %arg5, %arg10 = %arg6, %arg11 = %arg7) -> (tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>)  : i32 {
      %17 = arith.cmpi slt, %arg9, %cst_1 : tensor<128xi32>
      %18 = tt.addptr %5, %arg9 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %19 = tt.load %18, %17, %cst_0 : tensor<128x!tt.ptr<i32>>
      %20 = tt.load %arg11 : tensor<128x!tt.ptr<i32>>
      %21 = arith.addi %19, %20 : tensor<128xi32>
      tt.store %arg10, %21, %17 : tensor<128x!tt.ptr<i32>>
      %22 = arith.addi %arg9, %cst : tensor<128xi32>
      %23 = tt.addptr %arg10, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %24 = tt.addptr %arg11, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      scf.yield %22, %23, %24 : tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
    }
    %14 = arith.addi %13#0, %cst : tensor<128xi32>
    %15 = tt.addptr %13#1, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %16 = tt.addptr %13#2, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    scf.yield %14, %15, %16 : tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
  }
  tt.return
}

// CHECK-LABEL:   tt.func public @test_kernel(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %[[VAL_3:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<128> : tensor<128xi64>
// CHECK:           %[[VAL_5:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant dense<0> : tensor<128xi64>
// CHECK:           %[[VAL_9:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_12:.*]] = arith.constant dense<128> : tensor<128xi32>
// CHECK:           %[[VAL_13:.*]] = arith.constant dense<0> : tensor<128xi32>
// CHECK:           %[[VAL_14:.*]] = arith.constant dense<300> : tensor<128xi32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 128 : i32
// CHECK:           %[[VAL_16:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_16]], %[[VAL_15]] : i32
// CHECK:           %[[VAL_18:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK:           %[[VAL_19:.*]] = tt.splat %[[VAL_17]] : i32 -> tensor<128xi32>
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : tensor<128xi32>
// CHECK:           %[[VAL_21:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:           %[[VAL_22:.*]] = tt.splat %[[VAL_2]] : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:           %[[VAL_23:.*]] = tt.addptr %[[VAL_22]], %[[VAL_20]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
// CHECK:           %[[VAL_24:.*]] = tt.load %[[VAL_23]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[VAL_25:.*]] = arith.extsi %[[VAL_24]] : tensor<128xi32> to tensor<128xi64>
// CHECK:           %[[VAL_26:.*]]:3 = scf.for %[[VAL_27:.*]] = %[[VAL_11]] to %[[VAL_9]] step %[[VAL_10]] iter_args(%[[VAL_28:.*]] = %[[VAL_20]], %[[VAL_29:.*]] = %[[VAL_8]], %[[VAL_30:.*]] = %[[VAL_25]]) -> (tensor<128xi32>, tensor<128xi64>, tensor<128xi64>)  : i32 {
// CHECK:             %[[VAL_31:.*]] = tt.splat %[[VAL_1]] : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:             %[[VAL_32:.*]] = tt.addptr %[[VAL_31]], %[[VAL_29]] : tensor<128x!tt.ptr<i32>>, tensor<128xi64>
// CHECK:             %[[VAL_33:.*]]:3 = scf.for %[[VAL_34:.*]] = %[[VAL_11]] to %[[VAL_9]] step %[[VAL_10]] iter_args(%[[VAL_35:.*]] = %[[VAL_28]], %[[VAL_36:.*]] = %[[VAL_29]], %[[VAL_37:.*]] = %[[VAL_30]]) -> (tensor<128xi32>, tensor<128xi64>, tensor<128xi64>)  : i32 {
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_30]], %[[VAL_37]] : tensor<128xi64>
// CHECK:               %[[VAL_39:.*]] = tt.addptr %[[VAL_32]], %[[VAL_36]] : tensor<128x!tt.ptr<i32>>, tensor<128xi64>
// CHECK:               %[[VAL_40:.*]] = arith.cmpi slt, %[[VAL_35]], %[[VAL_14]] : tensor<128xi32>
// CHECK:               %[[VAL_41:.*]] = tt.addptr %[[VAL_21]], %[[VAL_35]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
// CHECK:               %[[VAL_42:.*]] = tt.load %[[VAL_41]], %[[VAL_40]], %[[VAL_13]] : tensor<128x!tt.ptr<i32>>
// CHECK:               %[[VAL_43:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:               %[[VAL_44:.*]] = scf.for %[[VAL_45:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_46:.*]] = %[[VAL_43]]) -> (tensor<128xi32>) {
// CHECK:                 %[[VAL_47:.*]] = tensor.extract %[[VAL_38]]{{\[}}%[[VAL_45]]] {DiscreteMemAccess} : tensor<128xi64>
// CHECK:                 %[[VAL_48:.*]] = tt.addptr %[[VAL_3]], %[[VAL_47]] : !tt.ptr<i32>, i64
// CHECK:                 %[[VAL_49:.*]] = tt.load %[[VAL_48]] {DiscreteMemAccess} : !tt.ptr<i32>
// CHECK:                 %[[VAL_50:.*]] = tt.splat %[[VAL_49]] : i32 -> tensor<1xi32>
// CHECK:                 %[[VAL_51:.*]] = tensor.insert_slice %[[VAL_50]] into %[[VAL_46]]{{\[}}%[[VAL_45]]] [1] [1] : tensor<1xi32> into tensor<128xi32>
// CHECK:                 scf.yield {DiscreteMemAccess} %[[VAL_51]] : tensor<128xi32>
// CHECK:               } {ExtractedLoadOrStore}
// CHECK:               %[[VAL_52:.*]] = arith.addi %[[VAL_42]], %[[VAL_44]] : tensor<128xi32>
// CHECK:               tt.store %[[VAL_39]], %[[VAL_52]], %[[VAL_40]] : tensor<128x!tt.ptr<i32>>
// CHECK:               %[[VAL_53:.*]] = arith.addi %[[VAL_35]], %[[VAL_12]] : tensor<128xi32>
// CHECK:               %[[VAL_54:.*]] = arith.addi %[[VAL_36]], %[[VAL_4]] : tensor<128xi64>
// CHECK:               %[[VAL_55:.*]] = arith.addi %[[VAL_37]], %[[VAL_4]] : tensor<128xi64>
// CHECK:               scf.yield %[[VAL_53]], %[[VAL_54]], %[[VAL_55]] : tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
// CHECK:             }
// CHECK:             %[[VAL_56:.*]] = arith.addi %[[VAL_30]], %[[VAL_57:.*]]#2 : tensor<128xi64>
// CHECK:             %[[VAL_58:.*]] = arith.addi %[[VAL_29]], %[[VAL_57]]#1 : tensor<128xi64>
// CHECK:             %[[VAL_59:.*]] = arith.addi %[[VAL_57]]#0, %[[VAL_12]] : tensor<128xi32>
// CHECK:             %[[VAL_60:.*]] = arith.addi %[[VAL_58]], %[[VAL_4]] : tensor<128xi64>
// CHECK:             %[[VAL_61:.*]] = arith.addi %[[VAL_56]], %[[VAL_4]] : tensor<128xi64>
// CHECK:             scf.yield %[[VAL_59]], %[[VAL_60]], %[[VAL_61]] : tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }

