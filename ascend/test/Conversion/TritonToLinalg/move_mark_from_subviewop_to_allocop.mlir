// RUN: triton-adapter-opt --triton-to-unstructure --bubble-up-operation --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @_fwd_kernel_alibi(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg5: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg6: f32 , %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg9: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg10: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg12: i32 {tt.divisibility = 16 : i32} , %arg13: i32, %arg14: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg15: i32 {tt.divisibility = 16 : i32} , %arg16: i32 {tt.divisibility = 16 : i32} , %arg17: i32 , %arg18: i32, %arg19: i32 , %arg20: i32 , %arg21: i32 , %arg22: i32 {tt.divisibility = 16 : i32} , %arg23: i32 , %arg24: i32 {tt.divisibility = 16 : i32} , %arg25: i32 {tt.divisibility = 16 : i32} , %arg26: i32 {tt.divisibility = 16 : i32} , %arg27: i32 , %arg28: i32 {tt.divisibility = 16 : i32} , %arg29: i32 {tt.divisibility = 16 : i32} , %arg30: i32 {tt.divisibility = 16 : i32} , %arg31: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<32xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_1 = arith.constant dense<0> : tensor<32xi64>
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %cst_2 = arith.constant dense<0xFF800000> : tensor<32x32xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_4 = arith.constant dense<0> : tensor<32xi32>
    %cst_5 = arith.constant dense<1> : tensor<32xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_6 = arith.constant dense<24> : tensor<32xi32>
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.divsi %1, %arg31 : i32
    %4 = tt.addptr %arg10, %0 : !tt.ptr<i64>, i32
    %5 = tt.load %4 : !tt.ptr<i64>
    %6 = tt.addptr %arg9, %0 : !tt.ptr<i64>, i32
    %7 = tt.load %6 : !tt.ptr<i64>
    %8 = tt.addptr %6, %c1_i32 : !tt.ptr<i64>, i32
    %9 = tt.load %8 : !tt.ptr<i64>
    %10 = arith.subi %9, %7 : i64
    %11 = arith.subi %5, %10 : i64
    %12 = arith.muli %2, %c32_i32 : i32
    %13 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %14 = tt.splat %12 : i32 ->tensor<32xi32>
    %15 = arith.addi %14, %13 : tensor<32xi32>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %17 = arith.extsi %16 : tensor<32x1xi32> to tensor<32x1xi64>
    %18 = tt.splat %7 : i64 -> tensor<32x1xi64>
    %19 = arith.addi %18, %17 : tensor<32x1xi64>
    %20 = arith.extsi %arg16 : i32 to i64
    %21 = tt.splat %20 : i64 -> tensor<32x1xi64>
    %22 = arith.muli %19, %21 : tensor<32x1xi64>
    %23 = arith.muli %1, %arg17 : i32
    %24 = arith.extsi %23 : i32 to i64
    %25 = tt.splat %24 : i64 -> tensor<32x1xi64>
    %26 = arith.addi %22, %25 : tensor<32x1xi64>
    %27 = tt.expand_dims %13 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %28 = arith.extsi %27 : tensor<1x32xi32> to tensor<1x32xi64>
    %29 = tt.broadcast %26 : tensor<32x1xi64> -> tensor<32x32xi64>
    %30 = tt.broadcast %28 : tensor<1x32xi64> -> tensor<32x32xi64>
    %31 = arith.addi %29, %30 : tensor<32x32xi64>
    %32 = arith.cmpi slt, %13, %cst_6 : tensor<32xi32>
    %33 = arith.select %32, %cst_5, %cst_4 : tensor<32xi1>, tensor<32xi32>
    %34 = arith.cmpi ne, %33, %cst_4 : tensor<32xi32>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %36 = arith.subi %5, %11 : i64
    %37 = tt.splat %36 : i64 -> tensor<32x1xi64>
    %38 = arith.cmpi slt, %17, %37 : tensor<32x1xi64>
    %39 = tt.broadcast %35 : tensor<1x32xi1> -> tensor<32x32xi1>
    %40 = tt.broadcast %38 : tensor<32x1xi1> -> tensor<32x32xi1>
    %41 = arith.andi %39, %40 : tensor<32x32xi1>
    %42 = tt.splat %arg0: !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %43 = tt.addptr %42, %31 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi64>
    %44 = tt.load %43, %41, %cst_3 : tensor<32x32x!tt.ptr<f32>>
    %45 = tt.addptr %arg11, %1 : !tt.ptr<f32>, i32
    %46 = tt.load %45 : !tt.ptr<f32>
    %47 = arith.extsi %15 : tensor<32xi32> to tensor<32xi64>
    %48 = tt.splat %11 : i64 ->tensor<32xi64>
    %49 = arith.addi %47, %48 : tensor<32xi64>
    %50 = arith.extsi %13 : tensor<32xi32> to tensor<32xi64>
    %51 = arith.muli %0, %arg15 : i32
    %52 = tt.addptr %arg5, %51 : !tt.ptr<i64>, i32
    %53 = arith.extsi %arg12 : i32 to i64
    %54 = tt.splat %53 : i64 -> tensor<32xi64>
    %55 = tt.splat %52 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>>
    %56 = arith.extsi %arg24 : i32 to i64
    %57 = tt.splat %56 : i64 -> tensor<1x32xi64>
    %58 = arith.muli %3, %arg25 : i32
    %59 = arith.extsi %58 : i32 to i64
    %60 = tt.splat %59 : i64 -> tensor<1x32xi64>
    %61 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %62 = tt.splat %arg13 : i32 -> tensor<32x1xi32>
    %63 = tt.splat %arg26 : i32 -> tensor<32x1xi32>
    %64 = tt.splat %53 : i64 -> tensor<1x32xi64>
    %65 = arith.extsi %arg27 : i32 to i64
    %66 = tt.splat %65 : i64 -> tensor<1x32xi64>
    %67 = arith.remsi %61, %62 : tensor<32x1xi32>
    %68 = arith.extsi %67 : tensor<32x1xi32> to tensor<32x1xi64>
    %69 = tt.broadcast %68 : tensor<32x1xi64> -> tensor<32x32xi64>
    %70 = arith.extsi %arg28 : i32 to i64
    %71 = tt.splat %70 : i64 -> tensor<32x1xi64>
    %72 = arith.muli %3, %arg29 : i32
    %73 = arith.extsi %72 : i32 to i64
    %74 = tt.splat %73 : i64 -> tensor<32x1xi64>
    %75 = tt.splat %arg30 : i32 -> tensor<1x32xi32>
    %76 = arith.muli %27, %75 : tensor<1x32xi32>
    %77 = arith.extsi %76 : tensor<1x32xi32> to tensor<1x32xi64>
    %78 = tt.broadcast %77 : tensor<1x32xi64> -> tensor<32x32xi64>
    %79 = arith.extsi %61 : tensor<32x1xi32> to tensor<32x1xi64>
    %80 = tt.splat %53 : i64 -> tensor<32x1xi64>
    %81 = tt.expand_dims %34 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
    %82 = tt.splat %11 : i64 -> tensor<1x32xi64>
    %83 = tt.broadcast %81 : tensor<32x1xi1> -> tensor<32x32xi1>
    %84 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %85 = tt.splat %arg6 : f32 -> tensor<32x32xf32>
    %86 = tt.expand_dims %49 {axis = 1 : i32} : tensor<32xi64> ->tensor<32x1xi64>
    %87 = tt.broadcast %86 : tensor<32x1xi64> -> tensor<32x32xi64>
    %88 = tt.splat %46 : f32 -> tensor<32x32xf32>
    %89 = tt.splat %5 : i64 -> tensor<32x1xi64>
    %90 = arith.cmpi slt, %86, %89 : tensor<32x1xi64>
    %91 = tt.broadcast %90 : tensor<32x1xi1> -> tensor<32x32xi1>
    %92 = tt.splat %11 : i64 -> tensor<32x1xi64>
    %93 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %94:4 = scf.for %arg32 = %c0_i64 to %11 step %c32_i64 iter_args(%arg33 = %c0_i32, %arg34 = %cst_3, %arg35 = %cst_0, %arg36 = %cst) -> (i32, tensor<32x32xf32>, tensor<32xf32>, tensor<32xf32>)  : i64 {
      %150 = tt.splat %arg32 : i64 -> tensor<32xi64>
      %151 = arith.addi %150, %50 : tensor<32xi64>
      %152 = arith.cmpi slt, %151, %48 : tensor<32xi64>
      %153 = arith.divsi %151, %54 : tensor<32xi64>
      %154 = tt.addptr %55, %153 : tensor<32x!tt.ptr<i64>>, tensor<32xi64>
      %155 = tt.load %154, %152, %cst_1 : tensor<32x!tt.ptr<i64>>
      %156 = tt.expand_dims %155 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
      %157 = arith.muli %156, %57 : tensor<1x32xi64>
      %158 = arith.addi %157, %60 : tensor<1x32xi64>
      %159 = arith.divsi %61, %62 : tensor<32x1xi32>
      %160 = arith.muli %159, %63 : tensor<32x1xi32>
      %161 = arith.extsi %160 : tensor<32x1xi32> to tensor<32x1xi64>
      %162 = tt.broadcast %158 : tensor<1x32xi64> -> tensor<32x32xi64>
      %163 = tt.broadcast %161 : tensor<32x1xi64> -> tensor<32x32xi64>
      %164 = arith.addi %162, %163 : tensor<32x32xi64>
      %165 = tt.splat %arg32 : i64 -> tensor<1x32xi64>
      %166 = arith.addi %165, %28 : tensor<1x32xi64>
      %167 = arith.remsi %166, %64 : tensor<1x32xi64>
      %168 = arith.muli %167, %66 : tensor<1x32xi64>
      %169 = tt.broadcast %168 : tensor<1x32xi64> ->tensor<32x32xi64>
      %170 = arith.addi %164, %169 : tensor<32x32xi64>
      %171 = arith.addi %170, %69 : tensor<32x32xi64>
      %172 = tt.expand_dims %155 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
      %173 = arith.muli %172, %71 : tensor<32x1xi64>
      %174 = arith.addi %173, %74 : tensor<32x1xi64>
      %175 = tt.broadcast %174 : tensor<32x1xi64> -> tensor<32x32xi64>
      %176 = arith.addi %175, %78 : tensor<32x32xi64>
      %177 = tt.splat %arg32 : i64 -> tensor<32x1xi64>
      %178 = arith.addi %177, %79 : tensor<32x1xi64>
      %179 = arith.remsi %178, %80 : tensor<32x1xi64>
      %180 = tt.broadcast %179 : tensor<32x1xi64> -> tensor<32x32xi64>
      %181 = arith.addi %176, %180 : tensor<32x32xi64>
      %182 = arith.cmpi slt, %166, %82 : tensor<1x32xi64>
      %183 = tt.broadcast %182 : tensor<1x32xi1> -> tensor<32x32xi1>
      %184 = arith.andi %83, %183 : tensor<32x32xi1>
      %185 = tt.addptr %84, %171 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi64>
      %186 = tt.load %185, %184, %cst_3 : tensor<32x32x!tt.ptr<f32>>
      %187 = tt.dot %44, %186, %cst_3 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>
      %188 = arith.select %183, %187, %cst_2 : tensor<32x32xi1>, tensor<32x32xf32>
      %189 = arith.mulf %188, %85 : tensor<32x32xf32>
      %190 = tt.splat %arg33 : i32 -> tensor<1x32xi32>
      %191 = arith.addi %27, %190 : tensor<1x32xi32>
      %192 = arith.extsi %191 : tensor<1x32xi32> to tensor<1x32xi64>
      %193 = tt.broadcast %192 : tensor<1x32xi64> -> tensor<32x32xi64>
      %194 = arith.subi %193, %87 : tensor<32x32xi64>
      %195 = arith.sitofp %194 : tensor<32x32xi64> to tensor<32x32xf32>
      %196 = arith.mulf %195, %88 : tensor<32x32xf32>
      %197 = arith.cmpf ole, %196, %cst_3 : tensor<32x32xf32>
      %198 = arith.andi %197, %91 : tensor<32x32xi1>
      %199 = arith.select %198, %196, %cst_2 : tensor<32x32xi1>, tensor<32x32xf32>
      %200 = arith.addf %189, %199 : tensor<32x32xf32>
      %201 = arith.addi %arg33, %c32_i32 : i32
      %202 = "tt.reduce"(%200) <{axis = 1 : i32}> ({
      ^bb0(%arg37: f32, %arg38: f32):
        %222 = arith.maxnumf %arg37, %arg38 : f32
        tt.reduce.return %222 : f32
      }) : (tensor<32x32xf32>) -> tensor<32xf32>
      %203 = arith.maxnumf %arg36, %202 : tensor<32xf32>
      %204 = tt.expand_dims %203 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
      %205 = tt.broadcast %204 : tensor<32x1xf32> -> tensor<32x32xf32>
      %206 = arith.subf %200, %205 : tensor<32x32xf32>
      %207 = math.exp %206 : tensor<32x32xf32>
      %208 = "tt.reduce"(%207) <{axis = 1 : i32}> ({
      ^bb0(%arg37: f32, %arg38: f32):
        %222 = arith.addf %arg37, %arg38 : f32
        tt.reduce.return %222 : f32
      }) : (tensor<32x32xf32>) -> tensor<32xf32>
      %209 = arith.subf %arg36, %203 : tensor<32xf32>
      %210 = math.exp %209 : tensor<32xf32>
      %211 = arith.mulf %210, %arg35 : tensor<32xf32>
      %212 = arith.addf %211, %208 : tensor<32xf32>
      %213 = tt.expand_dims %210 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
      %214 = tt.broadcast %213 : tensor<32x1xf32> -> tensor<32x32xf32>
      %215 = arith.mulf %arg34, %214 : tensor<32x32xf32>
      %216 = arith.cmpi slt, %178, %92 : tensor<32x1xi64>
      %217 = tt.broadcast %216 : tensor<32x1xi1> -> tensor<32x32xi1>
      %218 = arith.andi %39, %217 : tensor<32x32xi1>
      %219 = tt.addptr %93, %181 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi64>
      %220 = tt.load %219, %218, %cst_3 : tensor<32x32x!tt.ptr<f32>>
      %221 = tt.dot %207, %220, %215 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>
      scf.yield %201, %221, %212, %203 : i32, tensor<32x32xf32>, tensor<32xf32>, tensor<32xf32>
    } {tt.divisibility_arg1 = dense<32> : tensor<1xi32>}
    %95 = tt.splat %arg18 : i32 -> tensor<1x32xi32>
    %96 = arith.muli %27, %95 : tensor<1x32xi32>
    %97 = arith.muli %3, %arg19 : i32
    %98 = tt.splat %97 : i32 -> tensor<1x32xi32>
    %99 = arith.addi %96, %98 : tensor<1x32xi32>
    %100 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %101 = tt.broadcast %99 : tensor<1x32xi32> -> tensor<32x32xi32>
    %102 = tt.broadcast %100 : tensor<32x1xi32> -> tensor<32x32xi32>
    %103 = arith.addi %101, %102 : tensor<32x32xi32>
    %104 = tt.splat %arg20 : i32 -> tensor<32x1xi32>
    %105 = arith.muli %100, %104 : tensor<32x1xi32>
    %106 = arith.muli %3, %arg21 : i32
    %107 = tt.splat %106 : i32 ->tensor<32x1xi32>
    %108 = arith.addi %105, %107 : tensor<32x1xi32>
    %109 = tt.broadcast %108 : tensor<32x1xi32> -> tensor<32x32xi32>
    %110 = tt.broadcast %27 : tensor<1x32xi32> -> tensor<32x32xi32>
    %111 = arith.addi %109, %110 : tensor<32x32xi32>
    %112 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %113 = tt.addptr %112, %103 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %114 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %115 = tt.addptr %114, %111 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %116 = arith.extsi %12 : i32 to i64
    %117 = arith.cmpi slt, %116, %36 : i64
    %118 = arith.extui %117 : i1 to i32
    %119 = arith.addi %2, %c1_i32 : i32
    %120 = arith.muli %118, %119 : i32
    %121 = arith.muli %120, %c32_i32 : i32
    %122 = tt.expand_dims %34 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
    %123 = tt.splat %36 : i64 -> tensor<1x32xi64>
    %124 = tt.broadcast %122 : tensor<32x1xi1> ->tensor<32x32xi1>
    %125 = arith.extsi %arg18 : i32 to i64
    %126 = tt.splat %arg6 : f32 -> tensor<32x32xf32>
    %127 = tt.broadcast %16 : tensor<32x1xi32> -> tensor<32x32xi32>
    %128 = tt.expand_dims %49 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
    %129 = tt.broadcast %128 : tensor<32x1xi64> -> tensor<32x32xi64>
    %130 = tt.splat %46 : f32 -> tensor<32x32xf32>
    %131 = tt.splat %5 : i64 -> tensor<32x1xi64>
    %132 = arith.cmpi slt, %128, %131 : tensor<32x1xi64>
    %133 = tt.broadcast %132 : tensor<32x1xi1> -> tensor<32x32xi1>
    %134 = arith.extsi %arg20 : i32 to i64
    %135:4 = scf.for %arg32 = %c0_i32 to %121 step %c32_i32 iter_args(%arg33 = %11, %arg34 = %94#1, %arg35 = %94#2, %arg36 = %94#3) -> (i64, tensor<32x32xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
      %150 = tt.splat %arg32 : i32 -> tensor<1x32xi32>
      %151 = arith.addi %150, %27 : tensor<1x32xi32>
      %152 = arith.extsi %151 : tensor<1x32xi32> to tensor<1x32xi64>
      %153 = arith.cmpi slt, %152, %123 : tensor<1x32xi64>
      %154 = tt.broadcast %153 : tensor<1x32xi1> -> tensor<32x32xi1>
      %155 = arith.andi %124, %154 : tensor<32x32xi1>
      %156 = arith.extsi %arg32 : i32 to i64
      %157 = arith.addi %7, %156 : i64
      %158 = arith.muli %157, %125 : i64
      %159 = tt.splat %158 : i64 -> tensor<32x32xi64>
      %160 = tt.addptr %113, %159 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi64>
      %161 = tt.load %160, %155, %cst_3 : tensor<32x32x!tt.ptr<f32>>
      %162 = tt.dot %44, %161, %cst_3 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>
      %163 = arith.mulf %162, %126 : tensor<32x32xf32>
      %164 = tt.broadcast %151 : tensor<1x32xi32> -> tensor<32x32xi32>
      %165 = arith.cmpi sge, %127, %164 : tensor<32x32xi32>
      %166 = arith.select %165, %163, %cst_2 : tensor<32x32xi1>, tensor<32x32xf32>
      %167 = tt.splat %arg33 : i64 -> tensor<1x32xi64>
      %168 = arith.addi %28, %167 : tensor<1x32xi64>
      %169 = tt.broadcast %168 : tensor<1x32xi64> -> tensor<32x32xi64>
      %170 = arith.subi %169, %129 : tensor<32x32xi64>
      %171 = arith.sitofp %170 : tensor<32x32xi64> to tensor<32x32xf32>
      %172 = arith.mulf %171, %130 : tensor<32x32xf32>
      %173 = arith.cmpf ole, %172, %cst_3 : tensor<32x32xf32>
      %174 = arith.andi %173, %133 : tensor<32x32xi1>
      %175 = arith.select %174, %172, %cst_2 : tensor<32x32xi1>, tensor<32x32xf32>
      %176 = arith.addf %166, %175 : tensor<32x32xf32>
      %177 = arith.addi %arg33, %c32_i64 : i64
      %178 = "tt.reduce"(%176) <{axis = 1 : i32}> ({
      ^bb0(%arg37: f32, %arg38: f32):
          %203 = arith.maxnumf %arg37, %arg38 : f32
          tt.reduce.return %203 : f32
      }) : (tensor<32x32xf32>) -> tensor<32xf32>
      %179 = arith.maxnumf %arg36, %178 : tensor<32xf32>
      %180 = tt.expand_dims %179 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
      %181 = tt.broadcast %180 : tensor<32x1xf32> -> tensor<32x32xf32>
      %182 = arith.subf %176, %181 : tensor<32x32xf32>
      %183 = math.exp %182 : tensor<32x32xf32>
      %184 = "tt.reduce"(%183) <{axis = 1 : i32}> ({
      ^bb0(%arg37: f32, %arg38: f32):
          %203 = arith.addf %arg37, %arg38 : f32
          tt.reduce.return %203 : f32
      }) : (tensor<32x32xf32>) -> tensor<32xf32>
      %185 = arith.subf %arg36, %179 : tensor<32xf32>
      %186 = math.exp %185 : tensor<32xf32>
      %187 = arith.mulf %186, %arg35 : tensor<32xf32>
      %188 = arith.addf %187, %184 : tensor<32xf32>
      %189 = tt.expand_dims %186 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
      %190 = tt.broadcast %189 : tensor<32x1xf32> -> tensor<32x32xf32>
      %191 = arith.mulf %arg34, %190 : tensor<32x32xf32>
      %192 = tt.splat %arg32 : i32 -> tensor<32x1xi32>
      %193 = arith.addi %192, %100 : tensor<32x1xi32>
      %194 = arith.extsi %193 : tensor<32x1xi32> to tensor<32x1xi64>
      %195 = arith.cmpi slt, %194, %37 : tensor<32x1xi64>
      %196 = tt.broadcast %195 : tensor<32x1xi1> -> tensor<32x32xi1>
      %197 = arith.andi %39, %196 : tensor<32x32xi1>
      %198 = arith.muli %157, %134 : i64
      %199 = tt.splat %198 : i64 -> tensor<32x32xi64>
      %200 = tt.addptr %115, %199 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi64>
      %201 = tt.load %200, %197, %cst_3 : tensor<32x32x!tt.ptr<f32>>
      %202 = tt.dot %183, %201, %191 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>
      scf.yield %177, %202, %188, %179 : i64, tensor<32x32xf32>, tensor<32xf32>, tensor<32xf32>     
    } {tt.divisibility_arg1 = dense<32> : tensor<1xi32>}
    %136 = tt.expand_dims %135#2 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
    %137 = tt.broadcast %136 : tensor<32x1xf32> -> tensor<32x32xf32>
    %138 = arith.divf %135#1, %137 : tensor<32x32xf32>
    %139 = arith.extsi %arg22 : i32 to i64
    %140 = tt.splat %139 : i64 -> tensor<32x1xi64>
    %141 = arith.muli %19, %140 : tensor<32x1xi64>
    %142 = arith.muli %1, %arg23 : i32
    %143 = arith.extsi %142 : i32 to i64
    %144 = tt.splat %143 : i64 -> tensor<32x1xi64>
    %145 = arith.addi %141, %144 : tensor<32x1xi64>
    %146 = tt.broadcast %145 : tensor<32x1xi64> -> tensor<32x32xi64>
    %147 = arith.addi %146, %30 : tensor<32x32xi64>
    %148 = tt.splat %arg14 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %149 = tt.addptr %148, %147 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi64>
    tt.store %149, %138, %41 : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: %[[VAL_10:.*]] = memref.alloc() : memref<32x32xf32>
// CHECK: %[[VAL_24:.*]] = memref.alloc() : memref<32x32xf32>
// CHECK: annotation.mark %[[VAL_24]] {MayImplicitTransposeWithLastAxis} : memref<32x32xf32>
// CHECK: %[[VAL_16:.*]] = memref.alloc() : memref<32x32xf32>
// CHECK: annotation.mark %[[VAL_16]] {MayImplicitTransposeWithLastAxis} : memref<32x32xf32>
// CHECK: %[[VAL_27:.*]] = memref.alloc() : memref<32x32xf32>