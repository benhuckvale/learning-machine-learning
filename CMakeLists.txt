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

# Add your executable and include libtorch
add_executable(app main.cpp)
target_include_directories(app PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_directories(app PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(app PRIVATE torch)

set_property(TARGET app PROPERTY CXX_STANDARD 20)
