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

"""
Ascend-specific interpreter builder extensions.

This module extends the base InterpreterBuilder with Ascend-specific operations
(extension ops) without modifying the public base class. All Ascend-related
features are isolated here and can be extended independently.

Author: Triton-Ascend Contributors
"""

import numpy as np
import triton.language as tl
from .interpreter import InterpreterBuilder, TensorHandle


class AscendInterpreterBuilder(InterpreterBuilder):
    """
    Extended InterpreterBuilder with Ascend-specific extension operations.
    
    This class inherits from InterpreterBuilder and adds support for:
    - get_element (extract_scalar): Extract scalar from tensor using indices
    - insert_slice: Insert sub-tensor into full tensor
    - extract_slice: Extract slice from tensor
    - index_select_simd: SIMD gather operation
    - get_sub_vec_id: Get vector core ID for 1:2 ratio emulation
    - Synchronization operations: sync_block_set/wait/all
    
    All extension operations handle both TensorHandle and Python int types
    for interpreter mode compatibility.
    """
    
    def __init__(self) -> None:
        super().__init__()
        # Sub-vector core ID for simulating 1:2 hardware ratio
        self.sub_vec_id = 0
        # Flag to track if sub_vec_id simulation is needed
        self._sub_vec_simulation_enabled = False
    
    def get_additional_reserved_keywords(self):
        """
        Return additional reserved keywords specific to Ascend backend.
        
        These keywords will be filtered out from kernel call arguments
        and are not supported by the interpreter.
        
        :return: List of additional reserved keyword strings
        """
        return [
            "multibuffer",      # Ascend-specific memory buffering
            # Add more Ascend-specific keywords here as needed
            # "ascend_option1",
            # "ascend_option2",
        ]
    
    def patch_extensions(self, fn):
        """
        Patch Ascend extension modules for the given function.
        
        This method handles all Ascend-specific extension module patching,
        including CANN extensions and any other extension modules found in
        the function's global namespace.
        
        :param fn: The kernel function to patch extensions for
        """
        # Import _patch_builtin from parent module
        from .interpreter import _patch_builtin
        
        # Patch all modules in fn's globals that might be extension modules
        for name, value in list(fn.__globals__.items()):
            if value is None:
                continue
            try:
                # Check if it looks like an extension module (has builtin functions)
                if hasattr(value, '__name__') and 'extension' in str(value.__name__):
                    _patch_builtin(value, self)
                # Also try patching any module-like object that might have builtin functions
                elif hasattr(value, '__dict__') and not isinstance(value, type):
                    # Try to patch it and ignore if it fails
                    try:
                        _patch_builtin(value, self)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Also try importing extension directly as fallback
        try:
            import triton.language.extra.cann.extension as extension
            _patch_builtin(extension, self)
        except (ImportError, AttributeError):
            # Extension module not available (e.g., non-Ascend backend)
            pass
    
    def execute_with_sub_vec_simulation(self, fn, args, grid):
        """
        Execute function with optional 1:2 sub-vector core simulation.
        
        Sub-vector simulation is only activated when create_get_sub_vec_id() is
        actually called during execution. This avoids unnecessary double execution
        for code that doesn't use sub_vec_id functionality.
        
        :param fn: The kernel function to execute
        :param args: Function arguments
        :param grid: Grid dimensions (nx, ny, nz)
        """
        # Reset simulation flag at the beginning of each execution
        self._sub_vec_simulation_enabled = False
        self.sub_vec_id = 0
        
        # First, try a single execution to see if sub_vec_id is used
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    self.set_grid_idx(x, y, z)
                    fn(**args)
        
        # If sub_vec_id was accessed during execution, run again with sub_vec_id=1
        if self._sub_vec_simulation_enabled:
            self.sub_vec_id = 1
            for x in range(grid[0]):
                for y in range(grid[1]):
                    for z in range(grid[2]):
                        self.set_grid_idx(x, y, z)
                        fn(**args)

    # ========================================================================
    # Extension ops for Ascend
    # ========================================================================

    def create_extract_scalar(self, tensor_handle, indices):
        """
        Extract a scalar from a tensor using indices (equivalent to get_element).
        
        Handles mixed types: Python int (from loops) and TensorHandle (from other ops).
        
        :param tensor_handle: The tensor to extract from (TensorHandle)
        :param indices: List of scalar indices (can be TensorHandle or Python int)
        :return: Scalar value as TensorHandle
        """
        # Convert indices from TensorHandle or Python int to integers
        index_values = []
        for idx in indices:
            if isinstance(idx, int):
                # Python int passed directly (e.g., from loop counter)
                index_values.append(idx)
            elif isinstance(idx, TensorHandle):
                # Interpreter TensorHandle
                index_values.append(int(idx.data.item()) if hasattr(idx.data, 'item') else int(idx.data))
            else:
                # Fallback: try to extract data
                index_values.append(int(idx.data.item()) if hasattr(idx, 'data') and hasattr(idx.data, 'item') 
                                  else int(idx.data) if hasattr(idx, 'data') else int(idx))
        
        # Extract the scalar value
        scalar_data = tensor_handle.data[tuple(index_values)]
        return TensorHandle(np.array([scalar_data]), tensor_handle.dtype.scalar)

    def create_insert_slice(self, full_tensor, sub_tensor, offsets, sizes, strides):
        """
        Insert a sub-tensor into a full tensor at specified offsets.
        
        Handles mixed types: Python int and TensorHandle for offsets.
        
        :param full_tensor: The full tensor (destination, TensorHandle)
        :param sub_tensor: The sub-tensor to insert (TensorHandle)
        :param offsets: List of offset TensorHandle objects or Python ints
        :param sizes: List of size integers
        :param strides: List of stride integers
        :return: Modified tensor with sub_tensor inserted (TensorHandle)
        """
        result = full_tensor.data.copy()
        
        # Convert offsets from TensorHandle or Python int to integers
        offset_values = []
        for off in offsets:
            if isinstance(off, int):
                # Python int passed directly
                offset_values.append(off)
            elif isinstance(off, TensorHandle):
                # Interpreter TensorHandle
                offset_values.append(int(off.data.item()) if hasattr(off.data, 'item') else int(off.data))
            else:
                # Fallback
                offset_values.append(int(off.data.item()) if hasattr(off, 'data') and hasattr(off.data, 'item')
                                   else int(off.data) if hasattr(off, 'data') else int(off))
        
        # Build slices for insertion
        slices = []
        for i, (offset, size, stride) in enumerate(zip(offset_values, sizes, strides)):
            end = offset + size * stride
            if stride == 1:
                slices.append(slice(offset, end))
            else:
                slices.append(slice(offset, end, stride))
        
        # Insert the sub-tensor
        result[tuple(slices)] = sub_tensor.data
        
        return TensorHandle(result, full_tensor.dtype.scalar)

    def create_extract_slice(self, full_tensor, offsets, sizes, strides):
        """
        Extract a slice from a full tensor.
        
        Handles mixed types: Python int and TensorHandle for offsets.
        
        :param full_tensor: The full tensor (TensorHandle)
        :param offsets: List of offset TensorHandle objects or Python ints
        :param sizes: List of size integers
        :param strides: List of stride integers
        :return: Extracted sub-tensor (TensorHandle)
        """
        # Convert offsets from TensorHandle or Python int to integers
        offset_values = []
        for off in offsets:
            if isinstance(off, int):
                # Python int passed directly
                offset_values.append(off)
            elif isinstance(off, TensorHandle):
                # Interpreter TensorHandle
                offset_values.append(int(off.data.item()) if hasattr(off.data, 'item') else int(off.data))
            else:
                # Fallback
                offset_values.append(int(off.data.item()) if hasattr(off, 'data') and hasattr(off.data, 'item')
                                   else int(off.data) if hasattr(off, 'data') else int(off))
        
        # Build slices for extraction
        slices = []
        for i, (offset, size, stride) in enumerate(zip(offset_values, sizes, strides)):
            end = offset + size * stride
            if stride == 1:
                slices.append(slice(offset, end))
            else:
                slices.append(slice(offset, end, stride))
        
        # Extract the slice
        extracted = full_tensor.data[tuple(slices)]
        
        return TensorHandle(extracted, full_tensor.dtype.scalar)

    def create_index_select_simd(self, src_ptr, index_tensor, dim, src_shape, src_offset, read_shape, result_shape):
        """
        SIMD index_select operation (gather with indices along a dimension).
        
        This is a hardware-accelerated gather operation that selects elements
        from a tensor using a set of indices along a specified dimension.
        
        :param src_ptr: Source tensor pointer (TensorHandle)
        :param index_tensor: 1D tensor of indices (TensorHandle or array)
        :param dim: Dimension to select from (int)
        :param src_shape: List of source shape (int or TensorHandle)
        :param src_offset: List of source offset (int or TensorHandle)
        :param read_shape: List of read shape (int or TensorHandle)
        :param result_shape: List of result shape (int or TensorHandle)
        :return: Result tensor with selected indices (TensorHandle)
        """
        # Convert src_shape, src_offset, read_shape to integers
        def to_int(val):
            if isinstance(val, TensorHandle):
                return int(val.data.item())
            return int(val)
        
        src_shape_vals = [to_int(s) for s in src_shape]
        src_offset_vals = [to_int(o) if o != -1 else -1 for o in src_offset]
        read_shape_vals = [to_int(r) if r != -1 else -1 for r in read_shape]
        result_shape_vals = [to_int(r) for r in result_shape]
        
        # Get index values - handle both array and TensorHandle
        if isinstance(index_tensor, TensorHandle):
            indices = index_tensor.data.flatten()
        else:
            indices = np.asarray(index_tensor).flatten()
        
        # Ensure indices are integers
        if indices.dtype not in [np.int32, np.int64]:
            indices = indices.astype(np.int32)
        
        # Create result tensor
        result = np.empty(result_shape_vals, dtype=src_ptr.data.dtype)
        
        # Perform index_select: for each index, read the specified data
        for out_idx, in_idx in enumerate(indices):
            in_idx = int(in_idx)
            
            # Validate index bounds
            if not (0 <= in_idx < src_shape_vals[dim]):
                # Out of bounds - fill with zeros
                result_slices = [slice(None)] * len(result_shape_vals)
                result_slices[dim] = slice(out_idx, out_idx + 1)
                result[tuple(result_slices)] = 0
                continue
            
            # Build source slice
            src_slices = []
            for d in range(len(src_shape_vals)):
                if d == dim:
                    src_slices.append(slice(in_idx, in_idx + 1))
                else:
                    offset = src_offset_vals[d] if src_offset_vals[d] != -1 else 0
                    read_size = read_shape_vals[d] if read_shape_vals[d] != -1 else src_shape_vals[d]
                    # Clamp to valid range
                    offset = max(0, min(offset, src_shape_vals[d] - 1))
                    read_size = min(read_size, src_shape_vals[d] - offset)
                    src_slices.append(slice(offset, offset + read_size))
            
            # Build result slice
            result_slices = []
            for d in range(len(result_shape_vals)):
                if d == dim:
                    result_slices.append(slice(out_idx, out_idx + 1))
                else:
                    result_slices.append(slice(None))
            
            # Copy data with proper shape handling
            try:
                src_data = src_ptr.data[tuple(src_slices)]
                # Handle shape mismatch by resizing
                target_shape = [result_shape_vals[d] if d != dim else 1 for d in range(len(result_shape_vals))]
                if src_data.shape != tuple(target_shape):
                    # Pad or trim as needed
                    pad_width = [(0, target_shape[d] - src_data.shape[d]) for d in range(len(target_shape))]
                    src_data = np.pad(src_data, pad_width, mode='constant', constant_values=0)
                result[tuple(result_slices)] = src_data
            except Exception as e:
                # On error, fill with zeros
                result[tuple(result_slices)] = 0
        
        return TensorHandle(result, src_ptr.dtype.scalar)

    def create_get_sub_vec_id(self):
        """
        Get the Vector Core index on the AI Core.
        
        In Interpreter mode, simulate multiple vector cores by maintaining
        a sub_vec_id counter. This is used for 1:2 hardware ratio emulation
        where different vector cores process different partitions of the data.
        
        The first call to this method enables sub_vec_simulation, causing
        the kernel to be executed twice (once for each sub_vec_id value).
        
        :return: Vector Core ID as TensorHandle (int64, scalar)
        """
        # Enable sub_vec_id simulation when this method is called
        self._sub_vec_simulation_enabled = True
        
        # Return the current sub_vec_id
        vec_id = np.int64(self.sub_vec_id)
        return TensorHandle(np.array([vec_id], dtype=np.int64), tl.int64)

    def sync_block_set(self, sender, receiver, event_id, sender_pipe_value, receiver_pipe_value):
        """
        Set synchronization event between compute and vector units.
        
        In Interpreter mode, this is a no-op since we execute single-threaded.
        Synchronization is not needed in CPU emulation.
        
        :param sender: Source unit ("cube" or "vector")
        :param receiver: Destination unit ("cube" or "vector")
        :param event_id: Event ID (TensorHandle)
        :param sender_pipe_value: Sender pipe value
        :param receiver_pipe_value: Receiver pipe value
        """
        # No-op in interpreter mode: single-threaded execution doesn't need sync
        pass

    def sync_block_wait(self, sender, receiver, event_id, sender_pipe_value, receiver_pipe_value):
        """
        Wait for synchronization event between compute and vector units.
        
        In Interpreter mode, this is a no-op since we execute single-threaded.
        Synchronization is not needed in CPU emulation.
        
        :param sender: Source unit ("cube" or "vector")
        :param receiver: Destination unit ("cube" or "vector")
        :param event_id: Event ID (TensorHandle)
        :param sender_pipe_value: Sender pipe value
        :param receiver_pipe_value: Receiver pipe value
        """
        # No-op in interpreter mode: single-threaded execution doesn't need sync
        pass

    def sync_block_all(self, mode, event_id):
        """
        Synchronize all compute or vector units globally.
        
        In Interpreter mode, this is a no-op since we execute single-threaded.
        Synchronization is not needed in CPU emulation.
        
        :param mode: Sync mode ("all_cube", "all_vector", "all", "all_sub_vector")
        :param event_id: Event ID (int, constexpr, or TensorHandle)
        """
        # No-op in interpreter mode: single-threaded execution doesn't need sync
        pass
