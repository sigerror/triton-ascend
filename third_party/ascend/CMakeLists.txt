include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Triton Ascend is dependent on AscendNPU IR
set(ASCENDNPU_IR_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ascendnpu_ir")
set(ASCENDNPU_IR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ascendnpu_ir")
if(EXISTS ${ASCENDNPU_IR_SRC_DIR})
  message("Using existing AscendNPU-IR source directory: ${ASCENDNPU_IR_SRC_DIR}")
  set(FETCHCONTENT_SOURCE_DIR_ASCENDNPU_IR ${ASCENDNPU_IR_SRC_DIR})
else()
  message("AscendNPU-IR source directory does not exist. It will be fetched from remote.")
  unset(FETCHCONTENT_SOURCE_DIR_ASCENDNPU_IR)
endif()

set(ASCENDNPU_IR_TAG "master")  # Specify the desired tag or branch
if(NOT ${ASCENDNPU_IR_TAG})
  set(ASCENDNPU_IR_TAG "master")
endif()
include(FetchContent)
FetchContent_Declare(
  ascendnpu_ir
  GIT_REPOSITORY https://gitcode.com/Ascend/AscendNPU-IR.git
  GIT_TAG ${ASCENDNPU_IR_TAG}
  SOURCE_DIR ${ASCENDNPU_IR_SRC_DIR}
  BINARY_DIR ${ASCENDNPU_IR_BINARY_DIR}
  GIT_SUBMODULES ""
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

message("Start to fetch AscendNPU-IR repository.")
FetchContent_Populate(ascendnpu_ir)
# TODO: Check NPU IR version compatibility
FetchContent_GetProperties(ascendnpu_ir
  POPULATED is_ascendnpu_ir_populated
)
if (NOT is_ascendnpu_ir_populated)
  message(FATAL_ERROR "Failed to fetch AscendNPU-IR repository.")
endif()

set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)

add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})
include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)


add_triton_plugin(TritonAscend
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cc

  LINK_LIBS
  TritonToLinalg
)

# target_link_libraries(TritonAscend PRIVATE Python3::Module pybind11::headers)

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
