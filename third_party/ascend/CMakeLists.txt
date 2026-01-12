include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Triton Ascend is dependent on AscendNPU IR
set(ASCENDNPU_IR_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/AscendNPU-IR")
set(ASCENDNPU_IR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/AscendNPU-IR")

set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)

add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})
include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)

add_triton_plugin(TritonAscend
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ascend_ir.cc

  LINK_LIBS
  TritonToLinalg
  BiShengIRScopeDialect
  BiShengIRHIVMDialect
)

# target_link_libraries(TritonAscend PRIVATE Python3::Module pybind11::headers)

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
