# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import ast
from typing import Dict, List, Union


class AutoParser(ast.NodeVisitor):
    """
    Base class for parsing triton dsl kernel code using AST analysis.

    Provides common functionality for traversing the AST (abstract syntax tree)
    of a triton dsl kernel function and identifying specific elements in the code.

    Subclassed should implement specific parsing logic by overriding the relevant
    node visit methods. 
    """
    def __init__(self, func_ast: ast.AST):
        self.func_ast = func_ast

    def parse(self):
        self.visit(self.func_ast)
    
    def contains_target_var(self, node, var):
        """
        Recursively checks if a given AST node or its children contain a reference 
        to the specified variable.

        :param node: the AST node to check
        :type node: ast.AST
        :param var: the variable name to search for
        :type var: str
        :return: True if the variable is found, False otherwise
        """
        if isinstance(node, ast.Name) and node.id == var:
            return True
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if self.contains_target_var(item, var):
                        return True
            elif isinstance(value, ast.AST):
                if self.contains_target_var(value, var):
                    return True
        return False


class AxesKeyParser(AutoParser):
    """
    A parser for extracting axis information from a given function's AST.
    This class is designed to handle specific patterns in the function's code to
    determine the axis associated with a given variable. It is particularly useful
    for parsing triton DSL kernel code and identifying axis information.
    It recursively processes assignment nodes and lessthan nodes to obtain the axes
    corresponding to the specified var in the given function.
    """
    def __init__(self, func_ast: ast.AST, keys):
        super().__init__(func_ast)
        self.keys = keys
        self.checked_vars = list()

    def get_axis(self, var):
        axis = None
        for node in ast.walk(self.func_ast):
            # handle compare node
            if isinstance(node, ast.Compare):
                axis = self.handle_lt_node(node, var)
            elif isinstance(node, ast.Assign):
                axis = self.handle_assign_node(node, var)
            if axis is not None:
                return axis
        self.checked_vars.append(var)
        return None

    def handle_assign_node(self, node, var):
        if not isinstance(node, ast.Assign) or not isinstance(node.targets, list):
            return None
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None

        target = node.targets[0].id
        if target in self.checked_vars:
            return None
        # Prevent cyclic assignment.
        if var == target or not self.contains_target_var(node.value, var):
            return None
        
        axis = self.get_axis(target)
        if axis:
            return axis
        return None

    def handle_lt_node(self, node, var):
        if not isinstance(node, ast.Compare) or not isinstance(node.ops, list):
            return None
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Lt):
            return None
        if not isinstance(node.comparators, list) or len(node.comparators) != 1:
            return None
        if not isinstance(node.left, ast.Name) or var != node.left.id:
            return None
        
        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Name) and \
           not (isinstance(comparator, ast.Call) and \
           isinstance(comparator.func, ast.Name) and \
           comparator.func.id == 'min'):
            return None
        
        for k, v in self.keys.items():
            if self.contains_target_var(comparator, v):
                return k
        return None


class SplitAxesParser(AxesKeyParser):
    """
    Extracts the split axis parameters from triton kernel code. The parsing is based on the 
    `tl.program_id` statement. This class identifies potential split axes by analyzing the usage
    of the `tl.program_id` variable in the program and its multiplication operations with other
    variables(currently supporting scenarios where multiplication is either direct or indirect via
    intermediate variables). It then filters these candidates based on a list of candidate parameters
    (parameters not provided by the user). After that, it confirms the split axis corresponding to 
    the current parameter using mask comparison and the `keys` passed in `autotune`.
    
    Note:
    1. Split axis parameters must be multiplied with `tl.program_id`. 
    2. Without mask comparision, it is impossible to confirm the exact split axis, which would lead
       to parameter parsing failure. (eg. mask = offsets < n_elements)
    3. The identified split axes are limited to the list of candidated parameters, ensuring that
       only those parameters that can be dynamically adjusted through the autotune process are considered.
    """
    def __init__(self, func_ast: ast.AST, keys, candidates_params: List[str]):
        """
        :param func_ast: Abstract syntax tree of the triton kernel function
        :type func_ast: ast.AST
        :param keys: a dict of axis name: argument name, used to confirm the split axis corresponding to
            the split axis parameters.
        :type keys: Dict[str, str]
        :param candidates_params: a list of parameters names that were not provided by the user when calling
            triton kernel function. The parser will only consider these parameters as potential split axis
            parameters.
        :type candidates_params: List[str]
        """
        super().__init__(func_ast, keys)
        self.split_axes = dict()
        self.program_id_vars = list()
        self.candidates_params = candidates_params

    def parse(self) -> Dict[str, str]:
        super().parse()
        return self.split_axes
    
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if isinstance(node.value.func.value, ast.Name):
                if node.value.func.value.id == "tl" and node.value.func.attr == "program_id":
                    if isinstance(node.targets[0], ast.Name) and \
                       node.targets[0].id not in self.program_id_vars:
                        self.program_id_vars.append(node.targets[0].id)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mult):
            split_axes_val = None
            if isinstance(node.left, ast.Name) and node.left.id in self.program_id_vars:
                if isinstance(node.right, ast.Name):
                    split_axes_val = node.right.id
            elif isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Attribute):
                if node.left.func.value.id == "tl" and \
                   node.left.func.attr == "program_id":
                    if isinstance(node.right, ast.Name):
                        split_axes_val = node.right.id

            if isinstance(node.right, ast.Name) and node.right.id in self.program_id_vars:
                if isinstance(node.left, ast.Name):
                    split_axes_val = node.left.id
            elif isinstance(node.right, ast.Call) and isinstance(node.right.func, ast.Attribute):
                if node.right.func.value.id == "tl" and node.right.func.attr == "program_id":
                    if isinstance(node.left, ast.Name):
                        split_axes_val = node.left.id
            
            if split_axes_val in self.candidates_params and \
               split_axes_val not in self.split_axes.values():
                split_axes_key = self.get_axis(split_axes_val)
                if split_axes_key:
                    self.split_axes[split_axes_key] = split_axes_val
        self.generic_visit(node)


class TilingAxesParser(AxesKeyParser):
    """
    Extracts the tiling axis parameters from triton kernel code. The parsing is based on the 
    `tl.arange`, `tl.range` and `range()` statement. This class identifies potential tiling axes by analyzing
    the usage of the `range` and `tl.range` within `for` loop in the program. Common parameters
    between `range()` or `tl.range` and `tl.arange` are extracted. It then filters these candidates based on a 
    list of candidate parameters (parameters not provided by the user). After that, it confirms the
    tiling axis corresponding to the current parameter using mask comparison and the `keys` passed 
    in `autotune`.
    
    Note:
    1. Tiling axis parameters must be calculated within the `tl.arange` function and the `for` loop
       using `tl.range`. 
    2. Without mask comparision, it is impossible to confirm the exact tiling axis, which would lead
       to parameter parsing failure. (eg. mask = offsets < n_elements).
    3. The identified tiling axes are limited to the list of candidated parameters, ensuring that
       only those parameters that can be dynamically adjusted through the autotune process are considered.
    """
    def __init__(self, func_ast: ast.AST, keys, candidates_params: List[str]):
        """
        :param func_ast: Abstract syntax tree of the triton kernel function
        :type func_ast: ast.AST
        :param keys: a dict of axis name: argument name, used to confirm the tiling axis corresponding to
            the tiling axis parameters.
        :type keys: Dict[str, str]
        :param candidates_params: a list of parameters names that were not provided by the user when calling
            triton kernel function. The parser will only consider these parameters as potential tiling axis
            parameters.
        :type candidates_params: List[str]
        """
        super().__init__(func_ast, keys)
        self.tiling_axes = dict()
        self.candidates_params = candidates_params
        self.candidates_params_for_loop = list()
    
    def parse(self) -> Dict[str, str]:
        super().parse()
        return self.tiling_axes
    
    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and \
           len(node.iter.args) == 3 and \
           isinstance(node.iter.args[2], ast.Name):
            for_loop_param = node.iter.args[2].id
            if for_loop_param in self.candidates_params and \
               for_loop_param not in self.candidates_params_for_loop:
                self.candidates_params_for_loop.append(for_loop_param)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            # handle FloorDiv
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.FloorDiv):
                denominator = node.value.right
                if isinstance(denominator, ast.Name) and \
                   denominator.id in self.candidates_params and \
                   denominator.id not in self.candidates_params_for_loop:
                    self.candidates_params_for_loop.append(denominator.id)
                    self.visit(self.func_ast)

            tiling_axes_val = self.get_tiling_axes_val(node.value)
            if tiling_axes_val is not None and \
               tiling_axes_val in self.candidates_params_for_loop:
                tiling_axes_key = self.get_axis(tiling_axes_val)
                if tiling_axes_key:
                    self.tiling_axes[tiling_axes_key] = tiling_axes_val
        self.generic_visit(node)

    def get_tiling_axes_val(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'arange' and \
               isinstance(node.func.value, ast.Name) and \
               node.func.value.id == 'tl':
                if isinstance(node.args, list) and len(node.args) == 2:
                    return node.args[1].id

        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    val = self.get_tiling_axes_val(item)
                    if val:
                        return val
            elif isinstance(value, ast.AST):
                val = self.get_tiling_axes_val(value)
                if val:
                    return val
        return None


class LowDimsAxesParser(AutoParser):
    """
    Extracts the low dimensions axis from triton kernel code. The parsing is based on the 
    `tl.arange` statement. This class identifies low dimensions axis by analyzing the usage
    of the `tl.arange` in the program and extracts the variables computed by `tl.arange` and
    their associated operations. Then it checks if these variables are involved in slicing
    operations to determine dimension expansion and filters out variables that are expanded
    in non-lowest dimensions. After that, it compares the extracted variables with the provided
    `keys` to map them to specific low-dimensional axis.

    Note:
    1. low dimensions axis must be calculated within the `tl.arange` function and involved in 
       slicing operations to be identified.
    2. Without mask comparision, it is impossible to confirm the exact low dimensions axis, which
       would lead to parameter parsing failure. (eg. mask = offsets < n_elements).
    """
    def __init__(self, func_ast: ast.AST, keys: Dict[str, str]):
        """
        :param func_ast: Abstract syntax tree of the triton kernel function
        :type func_ast: ast.AST
        :param keys: a dict of axis name: argument name, used to confirm the low-dimensional axis.
        :type keys: Dict[str, str]
        """
        super().__init__(func_ast)
        self.low_dims_axis = list()
        self.keys = keys
        self.checked_compared_vars = list()
        self.checked_slice_vars = list()

    def parse(self):
        super().parse()
        return self.low_dims_axis

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            tl_arange_node = self.get_tl_arange_node(node)
            low_dims_axis = None
            if isinstance(tl_arange_node, ast.Call):
                partin_other_slice = [False]
                if self.is_partin_low_dim_slice(node.targets[0].id, partin_other_slice):
                    low_dims_axis = self.get_low_dims_axes(node.targets[0].id)
                if not partin_other_slice[0]:
                    low_dims_axis = self.get_low_dims_axes(node.targets[0].id)
            elif isinstance(tl_arange_node, ast.Subscript) and \
                 self.is_low_dim_slice(tl_arange_node, [False]):
                low_dims_axis = self.get_low_dims_axes(node.targets[0].id)

            if low_dims_axis and low_dims_axis not in self.low_dims_axis:
                self.low_dims_axis.append(low_dims_axis)
        self.generic_visit(node)

    def get_tl_arange_node(self, node):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if self.is_tl_arange_call(item):
                        return item
                    node = self.get_tl_arange_node(item)
                    if node:
                        return node
            elif isinstance(value, ast.AST):
                if self.is_tl_arange_call(value):
                    return value
                node = self.get_tl_arange_node(value)
                if node:
                    return node
        return None

    def is_tl_arange_call(self, node):
        """
        Checks if the given AST node is a call to `tl.arange` or a subscript of `tl.arange`. 
        It supports direct calls to `tl.arange` and subscripts of `tl.arange`, such as 
        `tl.arange()[None, :]`
        """
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'arange' and \
               isinstance(node.func.value, ast.Name) and \
               node.func.value.id == 'tl':
                return True
        elif isinstance(node, ast.Subscript):
            return self.is_tl_arange_call(node.value)
        return False
    
    def is_low_dim_slice(self, node: ast.Subscript, partin_other_slice):
        if not isinstance(node.slice, ast.Tuple) or not isinstance(node.slice.elts, list):
            return False
        elts = node.slice.elts
        if len(elts) != 0 and not isinstance(elts[-1], ast.Slice):
            partin_other_slice[0] = True
            return False
        return True
    
    def is_partin_low_dim_slice(self, var, partin_other_slice):
        for node in ast.walk(self.func_ast):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if var == node.value.id and self.is_low_dim_slice(node, partin_other_slice): 
                    return True
            elif isinstance(node, ast.Assign):
                if len(node.targets) == 1 and \
                   isinstance(node.targets[0], ast.Name) and \
                   var != node.targets[0].id: # Prevent cyclic assignment.
                    if not self.contains_target_var(node.value, var):
                        continue
                    target_var = node.targets[0].id
                    if target_var in self.checked_slice_vars:
                        continue
                    if self.is_partin_low_dim_slice(target_var, partin_other_slice):
                        return True

        self.checked_slice_vars.append(var)
        return False

    def get_low_dims_axes(self, var):
        for node in ast.walk(self.func_ast):
            if isinstance(node, ast.Compare):
                if not isinstance(node.left, ast.Name) or \
                   not isinstance(node.comparators[0], ast.Name):
                    continue
                if var == node.left.id:
                    compared_var = node.comparators[0].id
                elif var == node.comparators[0].id:
                    compared_var = node.left.id
                else:
                    continue
                for k, v in self.keys.items():
                    if v == compared_var:
                        return k
            elif isinstance(node, ast.Assign):
                if len(node.targets) == 1 and \
                   isinstance(node.targets[0], ast.Name) and \
                   var != node.targets[0].id: # Prevent cyclic assignment.
                    if not self.contains_target_var(node.value, var):
                        continue
                    target_var = node.targets[0].id
                    if target_var in self.checked_compared_vars:
                        continue
                    key = self.get_low_dims_axes(target_var)
                    if key is not None:
                        return key
        self.checked_compared_vars.append(var)
        return None


class PtrNumsParser(AutoParser):
    """
    Counts the number of pointer parameters from triton kernel code. The parsing of pointer-type
    parameters is determined based on whether these parameters participate in memory access
    statements such as `tl.load` and `tl.store`.
    First, all input parameters in the kernel function are parsed, and then recursively, all variables
    involved in the computation of each input parameter are identified.
    If an input parameter directly participates in the computation of the first argument of `tl.load`
    or `tl.store`, or if an intermediate variable computed from this input parameter indirectly
    participates in the computation of the first argument of `tl.load` or `tl.store`, then this
    parameter is considered a pointer-type parameter.

    Note:
    1. Variables modified with `tl.constexpr` are not pointer-type variables and will not be 
       further parsed.
    2. Only memory access statementes where the input parameter is directly involved or indirectly
       involved through one level of computation are counted. Intermediate variables computed from
       the input parameter through two or more levels of computation are not counted.
    """
    def __init__(self, func_ast: ast.AST, miss_params: List[str]):
        """
        :param func_ast: Abstract syntax tree of the triton kernel function
        :type func_ast: ast.AST
        :param miss_params: a list of parameters names that were not provided by the user when calling triton
            kernel function.
        :type miss_params: List[str]
        """
        super().__init__(func_ast)
        self.checked_vars = list()
        self.ptr_nums = 0
        self.ptr_params = list()
        self.miss_params = miss_params
        self.constexpr_params = list()

    def parse(self):
        super().parse()
        return self.ptr_nums, self.ptr_params
    
    def visit_FunctionDef(self, node):
        if isinstance(node.args, ast.arguments):
            for arg in node.args.args:
                if not isinstance(arg, ast.arg):
                    continue
                
                if isinstance(arg.annotation, ast.Attribute):
                    # var modified by tl.constexpr are not pointer type var, passed
                    is_tl = isinstance(arg.annotation.value, ast.Name) and \
                            arg.annotation.value.id == 'tl'
                    if is_tl and arg.annotation.attr == 'constexpr':
                        if arg.arg not in self.constexpr_params:
                            self.constexpr_params.append(arg.arg)
                        continue

                if self.is_in_addr_calc(arg.arg):
                    self.ptr_params.append(arg.arg)
                    self.ptr_nums += 1

        for miss_param in self.miss_params:
            if miss_param not in self.constexpr_params:
                print(f"[WARNING] The parameter '{miss_param}' needs to be declared as tl.constexpr!")
        self.generic_visit(node)
    
    def is_in_addr_calc(self, var):
        for node in ast.walk(self.func_ast):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and \
                   isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "tl" and \
                       (node.func.attr == "load" or node.func.attr == "store"):
                        if [arg for arg in node.args if self.contains_target_var(arg, var)]:
                            return True
                        
            elif isinstance(node, ast.Assign):
                if len(node.targets) == 1 and \
                   isinstance(node.targets[0], ast.Name) and \
                   var != node.targets[0].id: # Prevent cyclic assignment.
                    target_var = node.targets[0].id
                    if target_var in self.checked_vars:
                        continue
                    if isinstance(node.value, ast.BinOp) and \
                       isinstance(node.value.op, ast.Add):
                        if isinstance(node.value.left, ast.Name) and \
                           node.value.left.id == var:
                            if self.is_in_addr_calc(node.targets[0].id):
                                return True
                        elif isinstance(node.value.right, ast.Name) and \
                             node.value.right.id == var:
                            if self.is_in_addr_calc(node.targets[0].id):
                                return True
        self.checked_vars.append(var)
        return False
    
