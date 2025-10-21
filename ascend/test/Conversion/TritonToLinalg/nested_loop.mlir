// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

tt.func public @test_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<128> : tensor<128xi64>
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant dense<0> : tensor<128xi64>
  %c3_i32 = arith.constant 3 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst_1 = arith.constant dense<128> : tensor<128xi32>
  %cst_2 = arith.constant dense<0> : tensor<128xi32>
  %cst_3 = arith.constant dense<300> : tensor<128xi32>
  %c128_i32 = arith.constant 128 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c128_i32 : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
  %4 = arith.addi %3, %2 : tensor<128xi32>
  %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %6 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %7 = tt.addptr %6, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %8 = tt.load %7 : tensor<128x!tt.ptr<i32>>
  %9 = arith.extsi %8 : tensor<128xi32> to tensor<128xi64>
  %10:3 = scf.for %arg4 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg5 = %4, %arg6 = %cst_0, %arg7 = %9) -> (tensor<128xi32>, tensor<128xi64>, tensor<128xi64>)  : i32 {
    %11 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %12 = tt.addptr %11, %arg6 : tensor<128x!tt.ptr<i32>>, tensor<128xi64>
    %13:3 = scf.for %arg8 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg9 = %arg5, %arg10 = %arg6, %arg11 = %arg7) -> (tensor<128xi32>, tensor<128xi64>, tensor<128xi64>)  : i32 {
      %19 = tt.addptr %12, %arg10 : tensor<128x!tt.ptr<i32>>, tensor<128xi64>
      %20 = arith.cmpi slt, %arg9, %cst_3 : tensor<128xi32>
      %21 = tt.addptr %5, %arg9 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %22 = tt.load %21 : tensor<128x!tt.ptr<i32>>
      %23 = arith.select %20, %22, %cst_2 : tensor<128xi1>, tensor<128xi32>
      %24 = tensor.empty() : tensor<128xi32>
      %25 = scf.for %arg12 = %c0 to %c128 step %c1 iter_args(%arg13 = %24) -> (tensor<128xi32>) {
        %extracted = tensor.extract %arg7[%arg12] {DiscreteMemAccess} : tensor<128xi64>
        %extracted_4 = tensor.extract %arg11[%arg12] {DiscreteMemAccess} : tensor<128xi64>
        %32 = arith.addi %extracted, %extracted_4 : i64
        %33 = tt.addptr %arg3, %32 : !tt.ptr<i32>, i64
        %34 = tt.load %33 {DiscreteMemAccess} : !tt.ptr<i32>
        %35 = tt.splat %34 : i32 -> tensor<1xi32>
        %inserted_slice = tensor.insert_slice %35 into %arg13[%arg12] [1] [1] : tensor<1xi32> into tensor<128xi32>
        scf.yield {DiscreteMemAccess} %inserted_slice : tensor<128xi32>
      } {ExtractedLoadOrStore}
      %26 = arith.addi %23, %25 : tensor<128xi32>
      %27 = tt.load %19 : tensor<128x!tt.ptr<i32>>
      %28 = arith.select %20, %26, %27 : tensor<128xi1>, tensor<128xi32>
      tt.store %19, %28 {DiscreteMask} : tensor<128x!tt.ptr<i32>>
      %29 = arith.addi %arg9, %cst_1 : tensor<128xi32>
      %30 = arith.addi %arg10, %cst : tensor<128xi64>
      %31 = arith.addi %arg11, %cst : tensor<128xi64>
      scf.yield %29, %30, %31 : tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
    }
    %14 = arith.addi %arg7, %13#2 : tensor<128xi64>
    %15 = arith.addi %arg6, %13#1 : tensor<128xi64>
    %16 = arith.addi %13#0, %cst_1 : tensor<128xi32>
    %17 = arith.addi %15, %cst : tensor<128xi64>
    %18 = arith.addi %14, %cst : tensor<128xi64>
    scf.yield %16, %17, %18 : tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
  }
  tt.return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @test_kernel(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<?xi8>, %[[VAL_1:.*]]: memref<?xi8>,
// CHECK-SAME:                           %[[VAL_2:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %[[VAL_3:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %[[VAL_4:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %[[VAL_5:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
// CHECK-SAME:                           %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[VAL_12:.*]] = arith.constant 300 : i32
// CHECK:           %[[VAL_13:.*]] = arith.constant 128 : i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_16:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_17:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_19:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 128 : i64
// CHECK:           %[[VAL_21:.*]] = tensor.empty() : tensor<128xi64>
// CHECK:           %[[VAL_22:.*]] = linalg.fill ins(%[[VAL_20]] : i64) outs(%[[VAL_21]] : tensor<128xi64>) -> tensor<128xi64>
// CHECK:           %[[VAL_23:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[VAL_24:.*]] = linalg.fill ins(%[[VAL_14]] : i32) outs(%[[VAL_23]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:           %[[VAL_25:.*]] = linalg.fill ins(%[[VAL_12]] : i32) outs(%[[VAL_23]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:           %[[VAL_26:.*]] = arith.muli %[[VAL_9]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_23]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i32):
// CHECK:             %[[VAL_29:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : index to i32
// CHECK:             linalg.yield %[[VAL_30]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[VAL_31:.*]] = arith.index_cast %[[VAL_26]] : i32 to index
// CHECK:           %[[VAL_32:.*]] = memref.reinterpret_cast %[[VAL_4]] to offset: {{\[}}%[[VAL_31]]], sizes: [128], strides: [1] : memref<?xi32> to memref<128xi32, strided<[1], offset: ?>>
// CHECK:           %[[VAL_33:.*]] = memref.alloc() : memref<128xi32>
// CHECK:           memref.copy %[[VAL_32]], %[[VAL_33]] : memref<128xi32, strided<[1], offset: ?>> to memref<128xi32>
// CHECK:           %[[VAL_34:.*]] = bufferization.to_tensor %[[VAL_33]] restrict writable : memref<128xi32>
// CHECK:           %[[VAL_35:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_34]] : tensor<128xi32>) outs(%[[VAL_21]] : tensor<128xi64>) {
// CHECK:           ^bb0(%[[VAL_36:.*]]: i32, %[[VAL_37:.*]]: i64):
// CHECK:             %[[VAL_38:.*]] = arith.extsi %[[VAL_36]] : i32 to i64
// CHECK:             linalg.yield %[[VAL_38]] : i64
// CHECK:           } -> tensor<128xi64>
// CHECK:           %[[VAL_39:.*]]:4 = scf.for %[[VAL_40:.*]] = %[[VAL_14]] to %[[VAL_16]] step %[[VAL_15]] iter_args(%[[VAL_41:.*]] = %[[VAL_35]], %[[VAL_42:.*]] = %[[VAL_31]], %[[VAL_43:.*]] = %[[VAL_17]], %[[VAL_44:.*]] = %[[VAL_17]]) -> (tensor<128xi64>, index, index, index)  : i32 {
// CHECK:             %[[VAL_45:.*]]:3 = scf.for %[[VAL_46:.*]] = %[[VAL_14]] to %[[VAL_16]] step %[[VAL_15]] iter_args(%[[VAL_47:.*]] = %[[VAL_41]], %[[VAL_48:.*]] = %[[VAL_42]], %[[VAL_49:.*]] = %[[VAL_43]]) -> (tensor<128xi64>, index, index)  : i32 {
// CHECK:               %[[VAL_50:.*]] = arith.index_cast %[[VAL_48]] : index to i32
// CHECK:               %[[VAL_51:.*]] = linalg.fill ins(%[[VAL_50]] : i32) outs(%[[VAL_23]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:               %[[VAL_52:.*]] = linalg.fill ins(%[[VAL_15]] : i32) outs(%[[VAL_23]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:               %[[VAL_53:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_27]], %[[VAL_52]] : tensor<128xi32>, tensor<128xi32>) outs(%[[VAL_27]] : tensor<128xi32>) {
// CHECK:               ^bb0(%[[VAL_54:.*]]: i32, %[[VAL_55:.*]]: i32, %[[VAL_56:.*]]: i32):
// CHECK:                 %[[VAL_57:.*]] = arith.muli %[[VAL_54]], %[[VAL_55]] : i32
// CHECK:                 linalg.yield %[[VAL_57]] : i32
// CHECK:               } -> tensor<128xi32>
// CHECK:               %[[VAL_58:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_53]], %[[VAL_51]] : tensor<128xi32>, tensor<128xi32>) outs(%[[VAL_53]] : tensor<128xi32>) {
// CHECK:               ^bb0(%[[VAL_59:.*]]: i32, %[[VAL_60:.*]]: i32, %[[VAL_61:.*]]: i32):
// CHECK:                 %[[VAL_62:.*]] = arith.addi %[[VAL_59]], %[[VAL_60]] : i32
// CHECK:                 linalg.yield %[[VAL_62]] : i32
// CHECK:               } -> tensor<128xi32>
// CHECK:               %[[VAL_63:.*]] = arith.addi %[[VAL_43]], %[[VAL_49]] : index
// CHECK:               %[[VAL_64:.*]] = arith.addi %[[VAL_44]], %[[VAL_44]] : index
// CHECK:               %[[VAL_65:.*]] = memref.reinterpret_cast %[[VAL_3]] to offset: {{\[}}%[[VAL_63]]], sizes: [128], strides: {{\[}}%[[VAL_64]]] : memref<?xi32> to memref<128xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_66:.*]] = tensor.empty() : tensor<128xi1>
// CHECK:               %[[VAL_67:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_58]], %[[VAL_25]] : tensor<128xi32>, tensor<128xi32>) outs(%[[VAL_66]] : tensor<128xi1>) {
// CHECK:               ^bb0(%[[VAL_68:.*]]: i32, %[[VAL_69:.*]]: i32, %[[VAL_70:.*]]: i1):
// CHECK:                 %[[VAL_71:.*]] = arith.cmpi slt, %[[VAL_68]], %[[VAL_69]] : i32
// CHECK:                 linalg.yield %[[VAL_71]] : i1
// CHECK:               } -> tensor<128xi1>
// CHECK:               %[[VAL_72:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_48]]], sizes: [128], strides: {{\[}}%[[VAL_18]]] : memref<?xi32> to memref<128xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_73:.*]] = memref.alloc() : memref<128xi32>
// CHECK:               memref.copy %[[VAL_72]], %[[VAL_73]] : memref<128xi32, strided<[?], offset: ?>> to memref<128xi32>
// CHECK:               %[[VAL_74:.*]] = bufferization.to_tensor %[[VAL_73]] restrict writable : memref<128xi32>
// CHECK:               %[[VAL_75:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_67]], %[[VAL_74]], %[[VAL_24]] : tensor<128xi1>, tensor<128xi32>, tensor<128xi32>) outs(%[[VAL_74]] : tensor<128xi32>) {
// CHECK:               ^bb0(%[[VAL_76:.*]]: i1, %[[VAL_77:.*]]: i32, %[[VAL_78:.*]]: i32, %[[VAL_79:.*]]: i32):
// CHECK:                 %[[VAL_80:.*]] = arith.select %[[VAL_76]], %[[VAL_77]], %[[VAL_78]] : i32
// CHECK:                 linalg.yield %[[VAL_80]] : i32
// CHECK:               } -> tensor<128xi32>
// CHECK:               %[[VAL_81:.*]] = scf.for %[[VAL_82:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_18]] iter_args(%[[VAL_83:.*]] = %[[VAL_23]]) -> (tensor<128xi32>) {
// CHECK:                 %[[VAL_84:.*]] = tensor.extract %[[VAL_41]]{{\[}}%[[VAL_82]]] {DiscreteMemAccess} : tensor<128xi64>
// CHECK:                 %[[VAL_85:.*]] = tensor.extract %[[VAL_47]]{{\[}}%[[VAL_82]]] {DiscreteMemAccess} : tensor<128xi64>
// CHECK:                 %[[VAL_86:.*]] = arith.addi %[[VAL_84]], %[[VAL_85]] : i64
// CHECK:                 %[[VAL_87:.*]] = arith.index_cast %[[VAL_86]] : i64 to index
// CHECK:                 %[[VAL_88:.*]] = memref.reinterpret_cast %[[VAL_5]] to offset: {{\[}}%[[VAL_87]]], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:                 %[[VAL_89:.*]] = memref.load %[[VAL_88]]{{\[}}%[[VAL_17]]] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:                 %[[VAL_90:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:                 %[[VAL_91:.*]] = linalg.fill ins(%[[VAL_89]] : i32) outs(%[[VAL_90]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:                 %[[VAL_92:.*]] = tensor.insert_slice %[[VAL_91]] into %[[VAL_83]]{{\[}}%[[VAL_82]]] [1] [1] : tensor<1xi32> into tensor<128xi32>
// CHECK:                 scf.yield {DiscreteMemAccess} %[[VAL_92]] : tensor<128xi32>
// CHECK:               } {ExtractedLoadOrStore}
// CHECK:               %[[VAL_93:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_75]], %[[VAL_81]] : tensor<128xi32>, tensor<128xi32>) outs(%[[VAL_75]] : tensor<128xi32>) {
// CHECK:               ^bb0(%[[VAL_94:.*]]: i32, %[[VAL_95:.*]]: i32, %[[VAL_96:.*]]: i32):
// CHECK:                 %[[VAL_97:.*]] = arith.addi %[[VAL_94]], %[[VAL_95]] : i32
// CHECK:                 linalg.yield %[[VAL_97]] : i32
// CHECK:               } -> tensor<128xi32>
// CHECK:               %[[VAL_98:.*]] = memref.alloc() : memref<128xi32>
// CHECK:               memref.copy %[[VAL_65]], %[[VAL_98]] : memref<128xi32, strided<[?], offset: ?>> to memref<128xi32>
// CHECK:               %[[VAL_99:.*]] = bufferization.to_tensor %[[VAL_98]] restrict writable : memref<128xi32>
// CHECK:               %[[VAL_100:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_67]], %[[VAL_93]], %[[VAL_99]] : tensor<128xi1>, tensor<128xi32>, tensor<128xi32>) outs(%[[VAL_93]] : tensor<128xi32>) {
// CHECK:               ^bb0(%[[VAL_101:.*]]: i1, %[[VAL_102:.*]]: i32, %[[VAL_103:.*]]: i32, %[[VAL_104:.*]]: i32):
// CHECK:                 %[[VAL_105:.*]] = arith.select %[[VAL_101]], %[[VAL_102]], %[[VAL_103]] : i32
// CHECK:                 linalg.yield %[[VAL_105]] : i32
// CHECK:               } -> tensor<128xi32>
// CHECK:               bufferization.materialize_in_destination %[[VAL_100]] in writable %[[VAL_65]] : (tensor<128xi32>, memref<128xi32, strided<[?], offset: ?>>) -> ()
// CHECK:               %[[VAL_106:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_47]], %[[VAL_22]] : tensor<128xi64>, tensor<128xi64>) outs(%[[VAL_47]] : tensor<128xi64>) {
// CHECK:               ^bb0(%[[VAL_107:.*]]: i64, %[[VAL_108:.*]]: i64, %[[VAL_109:.*]]: i64):
// CHECK:                 %[[VAL_110:.*]] = arith.addi %[[VAL_107]], %[[VAL_108]] : i64
// CHECK:                 linalg.yield %[[VAL_110]] : i64
// CHECK:               } -> tensor<128xi64>
// CHECK:               %[[VAL_111:.*]] = arith.addi %[[VAL_48]], %[[VAL_19]] : index
// CHECK:               %[[VAL_112:.*]] = arith.addi %[[VAL_49]], %[[VAL_19]] : index
// CHECK:               scf.yield %[[VAL_106]], %[[VAL_111]], %[[VAL_112]] : tensor<128xi64>, index, index
// CHECK:             }
// CHECK:             %[[VAL_113:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_41]], %[[VAL_114:.*]]#0 : tensor<128xi64>, tensor<128xi64>) outs(%[[VAL_41]] : tensor<128xi64>) {
// CHECK:             ^bb0(%[[VAL_115:.*]]: i64, %[[VAL_116:.*]]: i64, %[[VAL_117:.*]]: i64):
// CHECK:               %[[VAL_118:.*]] = arith.addi %[[VAL_115]], %[[VAL_116]] : i64
// CHECK:               linalg.yield %[[VAL_118]] : i64
// CHECK:             } -> tensor<128xi64>
// CHECK:             %[[VAL_119:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_113]], %[[VAL_22]] : tensor<128xi64>, tensor<128xi64>) outs(%[[VAL_113]] : tensor<128xi64>) {
// CHECK:             ^bb0(%[[VAL_120:.*]]: i64, %[[VAL_121:.*]]: i64, %[[VAL_122:.*]]: i64):
// CHECK:               %[[VAL_123:.*]] = arith.addi %[[VAL_120]], %[[VAL_121]] : i64
// CHECK:               linalg.yield %[[VAL_123]] : i64
// CHECK:             } -> tensor<128xi64>
// CHECK:             %[[VAL_124:.*]] = arith.addi %[[VAL_125:.*]]#1, %[[VAL_19]] : index
// CHECK:             %[[VAL_126:.*]] = arith.addi %[[VAL_43]], %[[VAL_125]]#2 : index
// CHECK:             %[[VAL_127:.*]] = arith.addi %[[VAL_44]], %[[VAL_44]] : index
// CHECK:             %[[VAL_128:.*]] = arith.addi %[[VAL_126]], %[[VAL_19]] : index
// CHECK:             scf.yield %[[VAL_119]], %[[VAL_124]], %[[VAL_128]], %[[VAL_127]] : tensor<128xi64>, index, index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }

