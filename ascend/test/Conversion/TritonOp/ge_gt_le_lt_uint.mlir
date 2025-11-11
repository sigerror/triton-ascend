// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-a5=False force_simt_template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-a5=False' --split-input-file | FileCheck %s

module {
  tt.func public @triton_ge(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i8>>
    %8 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i8>>
    %11 = arith.cmpi uge, %7, %10 : tensor<4xi8>
    %12 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi8>
    tt.store %13, %14 : tensor<4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_ge
// CHECK:           %[[RES:.*]] = arith.cmpi uge, %[[X0:.*]], %[[X1:.*]] : tensor<4xi8>

// -----

module {
  tt.func public @triton_ge(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i16>, %arg2: !tt.ptr<i16>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i16>>
    %8 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i16>>
    %11 = arith.cmpi uge, %7, %10 : tensor<4xi16>
    %12 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi16>
    tt.store %13, %14 : tensor<4x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_ge
// CHECK:           %[[RES:.*]] = arith.cmpi uge, %[[X0:.*]], %[[X1:.*]] : tensor<4xi16>

// -----

module {
  tt.func public @triton_ge(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i32>>
    %11 = arith.cmpi uge, %7, %10 : tensor<4xi32>
    %12 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi32>
    tt.store %13, %14 : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_ge
// CHECK:           %[[RES:.*]] = arith.cmpi uge, %[[X0:.*]], %[[X1:.*]] : tensor<4xi32>

// -----

module {
  tt.func public @triton_ge(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<i64>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i64>>
    %8 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i64>>
    %11 = arith.cmpi uge, %7, %10 : tensor<4xi64>
    %12 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi64>
    tt.store %13, %14 : tensor<4x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_ge
// CHECK:           %[[RES:.*]] = arith.cmpi uge, %[[X0:.*]], %[[X1:.*]] : tensor<4xi64>

// -----

module {
  tt.func public @triton_gt(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i8>>
    %8 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i8>>
    %11 = arith.cmpi ugt, %7, %10 : tensor<4xi8>
    %12 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi8>
    tt.store %13, %14 : tensor<4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_gt
// CHECK:           %[[RES:.*]] = arith.cmpi ugt, %[[X0:.*]], %[[X1:.*]] : tensor<4xi8>

// -----

module {
  tt.func public @triton_gt(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i16>, %arg2: !tt.ptr<i16>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i16>>
    %8 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i16>>
    %11 = arith.cmpi ugt, %7, %10 : tensor<4xi16>
    %12 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi16>
    tt.store %13, %14 : tensor<4x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_gt
// CHECK:           %[[RES:.*]] = arith.cmpi ugt, %[[X0:.*]], %[[X1:.*]] : tensor<4xi16>

// -----

module {
  tt.func public @triton_gt(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i32>>
    %11 = arith.cmpi ugt, %7, %10 : tensor<4xi32>
    %12 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi32>
    tt.store %13, %14 : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_gt
// CHECK:           %[[RES:.*]] = arith.cmpi ugt, %[[X0:.*]], %[[X1:.*]] : tensor<4xi32>

// -----

module {
  tt.func public @triton_gt(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<i64>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i64>>
    %8 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i64>>
    %11 = arith.cmpi ugt, %7, %10 : tensor<4xi64>
    %12 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi64>
    tt.store %13, %14 : tensor<4x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_gt
// CHECK:           %[[RES:.*]] = arith.cmpi ugt, %[[X0:.*]], %[[X1:.*]] : tensor<4xi64>

// -----

module {
  tt.func public @triton_le(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i8>>
    %8 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i8>>
    %11 = arith.cmpi ule, %7, %10 : tensor<4xi8>
    %12 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi8>
    tt.store %13, %14 : tensor<4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_le
// CHECK:           %[[RES:.*]] = arith.cmpi ule, %[[X0:.*]], %[[X1:.*]] : tensor<4xi8>

// -----

module {
  tt.func public @triton_le(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i16>, %arg2: !tt.ptr<i16>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i16>>
    %8 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i16>>
    %11 = arith.cmpi ule, %7, %10 : tensor<4xi16>
    %12 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi16>
    tt.store %13, %14 : tensor<4x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_le
// CHECK:           %[[RES:.*]] = arith.cmpi ule, %[[X0:.*]], %[[X1:.*]] : tensor<4xi16>

// -----

module {
  tt.func public @triton_le(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i32>>
    %11 = arith.cmpi ule, %7, %10 : tensor<4xi32>
    %12 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi32>
    tt.store %13, %14 : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_le
// CHECK:           %[[RES:.*]] = arith.cmpi ule, %[[X0:.*]], %[[X1:.*]] : tensor<4xi32>

// -----

module {
  tt.func public @triton_le(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<i64>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i64>>
    %8 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i64>>
    %11 = arith.cmpi ule, %7, %10 : tensor<4xi64>
    %12 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi64>
    tt.store %13, %14 : tensor<4x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_le
// CHECK:           %[[RES:.*]] = arith.cmpi ule, %[[X0:.*]], %[[X1:.*]] : tensor<4xi64>

// -----

module {
  tt.func public @triton_lt(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i8>>
    %8 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i8>>
    %11 = arith.cmpi ult, %7, %10 : tensor<4xi8>
    %12 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi8>
    tt.store %13, %14 : tensor<4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_lt
// CHECK:           %[[RES:.*]] = arith.cmpi ult, %[[X0:.*]], %[[X1:.*]] : tensor<4xi8>

// -----

module {
  tt.func public @triton_lt(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i16>, %arg2: !tt.ptr<i16>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i16>>
    %8 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i16>>
    %11 = arith.cmpi ult, %7, %10 : tensor<4xi16>
    %12 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi16>
    tt.store %13, %14 : tensor<4x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_lt
// CHECK:           %[[RES:.*]] = arith.cmpi ult, %[[X0:.*]], %[[X1:.*]] : tensor<4xi16>

// -----

module {
  tt.func public @triton_lt(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i32>>
    %11 = arith.cmpi ult, %7, %10 : tensor<4xi32>
    %12 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi32>
    tt.store %13, %14 : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_lt
// CHECK:           %[[RES:.*]] = arith.cmpi ult, %[[X0:.*]], %[[X1:.*]] : tensor<4xi32>

// -----

module {
  tt.func public @triton_lt(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<i64>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i64>>
    %8 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i64>>
    %11 = arith.cmpi ult, %7, %10 : tensor<4xi64>
    %12 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %13 = tt.addptr %12, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %14 = arith.extui %11 : tensor<4xi1> to tensor<4xi64>
    tt.store %13, %14 : tensor<4x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_lt
// CHECK:           %[[RES:.*]] = arith.cmpi ult, %[[X0:.*]], %[[X1:.*]] : tensor<4xi64>