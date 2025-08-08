cmake_minimum_required(VERSION 3.18)
project(UCP LANGUAGES CXX)

# === Try to find CUDA ===
include(CheckLanguage)
check_language(CUDA)

# === Try to find HIP ===
find_package(HIP QUIET)

# === Try to find MPI ===
find_package(MPI QUIET)

# === CUDA build ===
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "🔷 CUDA detected, building CUDA version")
    add_executable(${PROJECT_NAME}_cuda src/UCP_cuda.cu)
    set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 14
        CUDA_STANDARD 14)
    target_compile_options(${PROJECT_NAME}_cuda PRIVATE -O2)

# === HIP build ===
elseif(HIP_FOUND)
    message(STATUS "🟣 HIP detected, building HIP version")

    # Explicitly set hipcc as compiler
    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
    enable_language(HIP)

    add_executable(${PROJECT_NAME}_hip src/UCP_hip.cpp)
    target_include_directories(${PROJECT_NAME}_hip PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}_hip PRIVATE ${HIP_LIBRARIES})
    target_compile_options(${PROJECT_NAME}_hip PRIVATE -O2)

endif()

if(MPI_FOUND)
    message(STATUS "🟢 MPI detected, building MPI version")
    add_executable(${PROJECT_NAME}_mpi src/UCP_mpi.cpp)
    target_include_directories(${PROJECT_NAME}_mpi PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME}_mpi PRIVATE ${MPI_LIBRARIES})
    target_compile_options(${PROJECT_NAME}_mpi PRIVATE -O2)
endif()

