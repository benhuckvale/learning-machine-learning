cmake_minimum_required(VERSION 3.24)

project(learning-machine-learning)

include(FetchContent)

# Set the URL and file name of the libtorch zip file
# A URL to download the latest version of libtorch can be found at
# https://pytorch.org/get-started/locally/
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.1.zip")
set(LIBTORCH_FILENAME "libtorch.zip")

# Download the libtorch zip file using FetchContent
FetchContent_Declare(
  libtorch
  URL ${LIBTORCH_URL}
  DOWNLOAD_NAME ${LIBTORCH_FILENAME}
)

FetchContent_GetProperties(libtorch)
if(NOT libtorch_POPULATED)
  FetchContent_Populate(libtorch)
  set(TORCH_INCLUDE_DIRS ${libtorch_SOURCE_DIR}/include ${libtorch_SOURCE_DIR}/include/torch/csrc/api/include)
  set(TORCH_LIBRARIES ${libtorch_SOURCE_DIR}/lib)
endif()

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/onqtam/doctest.git
  GIT_TAG        2.4.4
)
FetchContent_MakeAvailable(doctest)

set(EXES
  00_simple_1x1_forward_pass
  01_simple_perceptron
  02_flower_perceptron
  03_handrolled_forward_pass
  06_handrolled_single_neuron
)

foreach(EXE ${EXES})
  add_executable(${EXE} src/${EXE}.cpp)
  target_include_directories(${EXE} PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_directories(${EXE} PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(${EXE} PRIVATE torch c10 torch_cpu doctest)
  set_property(TARGET ${EXE} PROPERTY CXX_STANDARD 20)
endforeach()
