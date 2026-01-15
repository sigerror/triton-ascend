try:
    import acl
    is_compile_on_910_95 = acl.get_soc_name().startswith("Ascend910_95")
except Exception as e:
    is_compile_on_910_95 = False

from .core import (
    builtin,
    is_builtin,
    ascend_address_space,
    sub_vec_id,
    copy_from_ub_to_l1,
    sync_block_set,
    sync_block_wait,
    PIPE,
    fixpipe,
    FixpipeDualDstMode,
    FixpipeDMAMode,
    FixpipePreQuantMode,
    FixpipePreReluMode,
    sync_block_all,
)

from .scope import scope


from .math_ops import (
    atan2,
    isfinited,
    finitef
)

from .aux_ops import (
    parallel,
    compile_hint,
    multibuffer,
)

from .vec_ops import (
    insert_slice,
    extract_slice,
    get_element,
    sort,
    flip,
    cast,
)

from .mem_ops import (
    index_select,
    index_put,
    gather_out_to_ub,
    scatter_ub_to_out,
    index_select_simd,
)

__all__ = [
    # core
    "builtin",
    "is_builtin",
    "sub_vec_id",
    "copy_from_ub_to_l1",
    "PIPE",
    "FixpipeDMAMode",
    "FixpipeDualDstMode",
    "FixpipePreQuantMode",
    "FixpipePreReluMode",
    "fixpipe",
    "sync_block_all",

    # address space
    "ascend_address_space",

    # scope
    "scope",

    # math ops
    "atan2",
    "isfinited",
    "finitef",

    # aux ops
    "sync_block_set",
    "sync_block_wait",
    "parallel",
    "compile_hint",
    "multibuffer",

    # vec ops
    "insert_slice",
    "extract_slice",
    "get_element",
    "sort",
    "flip",
    "cast",

    # mem ops
    "index_select",
    "index_put",
    "gather_out_to_ub",
    "scatter_ub_to_out",
    "index_select_simd",
]
