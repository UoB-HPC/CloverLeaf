register_flag_optional(RAJA_BACK_END "Specify whether we target CPU/CUDA/HIP/SYCL" "CPU")

register_flag_optional(MANAGED_ALLOC "Use UVM (cudaMallocManaged) instead of the device-only allocation (cudaMalloc)"
        "OFF")

register_flag_optional(SYNC_ALL_KERNELS
        "Fully synchronise all kernels after launch, this also enables synchronous error checking with line and file name"
        "OFF")

macro(setup)

    set(CMAKE_CXX_STANDARD 17)

    find_package(RAJA REQUIRED)
    find_package(umpire REQUIRED)
    find_package(Threads REQUIRED)

    register_link_library(RAJA umpire)
    if (${RAJA_BACK_END} STREQUAL "CUDA")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -extended-lambda --expt-relaxed-constexpr --restrict --keep -use_fast_math")

        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE CUDA)
        register_definitions(RAJA_TARGET_GPU)
    elseif (${RAJA_BACK_END} STREQUAL "HIP")
        find_package(hip REQUIRED)

        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 17)

        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE HIP)

        register_definitions(RAJA_TARGET_GPU)
    elseif (${RAJA_BACK_END} STREQUAL "SYCL")
        register_definitions(RAJA_TARGET_GPU)
    else()
        register_definitions(RAJA_TARGET_CPU)
        message(STATUS "Falling Back to CPU")
    endif ()   
    
    if (MANAGED_ALLOC)
        register_definitions(CLOVER_MANAGED_ALLOC)
    endif ()

    if (SYNC_ALL_KERNELS)
        register_definitions(CLOVER_SYNC_ALL_KERNELS)
    endif ()

endmacro()
