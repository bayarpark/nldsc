cmake_minimum_required(VERSION 3.20)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<0:>${CMAKE_BINARY_DIR}")


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# compile flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(_ldscore)

set(CMAKE_CXX_STANDARD 17)


# armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
set(GCC_COMPILE_FLAGS "-llapack -lblas -march=native -larmadillo -fopenmp -fno-fast-math")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}")


# pybind11
set(PYBIND11_DIR "./pybind11")
add_subdirectory(${PYBIND11_DIR} ./pybind11)
pybind11_add_module(${PROJECT_NAME} ldscore.cpp)

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        include
        ${ARMADILLO_LIBRARIES}
)


# Next line is required for linking both Armadillo, BLAS and LAPACK
target_link_libraries(${PROJECT_NAME} PRIVATE armadillo blas m lapack)
target_include_directories(${PROJECT_NAME} PRIVATE ${ARMADILLO_LIBRARIES})
