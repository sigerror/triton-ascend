/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_LINERIZE_ANALYSIS_OPFOLDRESULT_UTILS_H
#define TRITON_LINERIZE_ANALYSIS_OPFOLDRESULT_UTILS_H

#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"

#include <optional>

namespace mlir {

class OpBuilder;

// Return if ofr contains a constant zero, either represented by an integer
// attribute or a constant value.
bool hasConstZero(const OpFoldResult ofr);

// Create a value of index type if necessary from an OpFoldResult.
Value ofrToIndexValue(const OpFoldResult ofr, const Location loc, OpBuilder &b);

// Create a vector of values of index type if necessary from an array of
// OpFoldResults.
SmallVector<Value> ofrsToIndexValues(ArrayRef<OpFoldResult> ofrs,
                                     const Location loc, OpBuilder &b);

// Process addition of two OFRs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult addOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);

// Produce result = lhs - rhs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult subOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);

// Process multiplication of two OFRs. If both OFRs are Integer Attributes,
// result is an Integer Attribtue. Otherwise, insert the arith.muli
// instruction if needed and use its result Value.
OpFoldResult mulOFRValue(const OpFoldResult lhs, const Value rhs,
                         const Location loc, OpBuilder &b);

// Process division of two OFRs. If both OFRs are Integer Attributes,
// the result is an Integer Attribute. Otherwise, insert the arith.divi
// instruction if needed and use its result Value.
OpFoldResult divOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);

OpFoldResult remOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);


OpFoldResult mulOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);

OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);

OpFoldResult maxOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);
} // namespace mlir

#endif
