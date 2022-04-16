cmake_minimum_required(VERSION 3.20)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<0:>${CMAKE_BINARY_DIR}")

project(ldscore)

set(CMAKE_CXX_STANDARD 11)

set(PYBIND11_DIR "pybind11")
set(ARMA_COMPILE "-DARMA_DONT_USE_WRAPPER -larmadillo")
set(LAPACK "-llapack")
set(BLAS "-lblas")

add_definitions(
        ${LAPACK},
        ${BLAS},
        ${ARMA_COMPILE}
)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})



add_subdirectory(${PYBIND11_DIR} pybind11)

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