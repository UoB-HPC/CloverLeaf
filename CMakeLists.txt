cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

project(cloverleaf VERSION 1.0 LANGUAGES C CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)


## Flags for debugging only, enable for development (ASan only works on few models)
set(SANITIZE OFF)
if (SANITIZE)
    set(CXX_EXTRA_FLAGS -fsanitize=address -fsanitize=undefined ${CXX_EXTRA_FLAGS})
    set(CXX_EXTRA_LINK_FLAGS -fsanitize=address -fsanitize=undefined ${CXX_EXTRA_LINK_FLAGS})
endif ()


# the final executable name
set(EXE_NAME cloverleaf)


## select default build type if not given
if (NOT CMAKE_BUILD_TYPE)
    message("No CMAKE_BUILD_TYPE specified, defaulting to 'Release'")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ((NOT BUILD_TYPE STREQUAL RELEASE) AND (NOT BUILD_TYPE STREQUAL DEBUG))
    message(FATAL_ERROR "Only Release or Debug is supported, got `${CMAKE_BUILD_TYPE}`")
endif ()

# setup some defaults flags for everything
set(DEFAULT_DEBUG_FLAGS -Wall -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -O2)
set(DEFAULT_RELEASE_FLAGS -Wall -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -O3)

macro(hint_flag FLAG DESCRIPTION)
    if (NOT DEFINED ${FLAG})
        message(STATUS "${FLAG}: ${DESCRIPTION}")
    else ()
        # i.e. `-DFOO="-a -b"` becomes CMake's semicolon separated list `FOO=`-a;-b`
        separate_arguments(${FLAG})
    endif ()
endmacro()

# hint common extra flag options for all models if they are not set
hint_flag(CXX_EXTRA_FLAGS "
        Appends to common compile flags. These will be appended at link phase as well.
        To use separate flags at link phase, set `CXX_EXTRA_LINK_FLAGS`")
hint_flag(CXX_EXTRA_LINK_FLAGS "
        Appends to link flags which appear *before* the objects.
        Do not use this for linking libraries, as the link line is order-dependent")
hint_flag(CXX_EXTRA_LIBRARIES "
        Append to link flags which appear *after* the objects.
        Use this for linking extra libraries (e.g `-lmylib`, or simply `mylib`)")
hint_flag(CXX_EXTRA_LINKER_FLAGS "
        Append to linker flags (i.e GCC's `-Wl` or equivalent)")

# copy CXX_EXTRA_FLAGS <- CXX_EXTRA_LINK_FLAGS
if ((DEFINED CXX_EXTRA_FLAGS) AND (NOT DEFINED CXX_EXTRA_LINK_FLAGS))
    set(CXX_EXTRA_LINK_FLAGS ${CXX_EXTRA_FLAGS})
endif ()

option(USE_ONETBB "Enable oneTBB library for *supported* models. Enabling this on models that
                don't explicitly link against TBB is a no-op, see description of your selected
                model on how this is used." OFF)

option(FETCH_ONETBB "Fetch (download) the oneTBB library for *supported* models. This uses CMake's
                  FetchContent feature. Specify version by setting FETCH_ONETBB_VERSION" OFF)
set(FETCH_ONETBB_VERSION "v2021.13.0" CACHE STRING "Specify version of oneTBB to use if FETCH_ONETBB is ON")

if (FETCH_ONETBB)
    FetchContent_Declare(
            TBB
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG "${FETCH_ONETBB_VERSION}"
    )
    # Don't fail builds on waring (TBB has -Wall while not being free of warnings from unused symbols...)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(TBB_STRICT OFF)
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(TBB)
    if (NOT TBB_POPULATED)
        FetchContent_Populate(TBB)
        add_subdirectory(${tbb_SOURCE_DIR} ${tbb_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()

option(USE_ONEDPL "Enable oneDPL library for *supported* models. Enabling this on models that
                   don't explicitly link against DPL is a no-op, see description of your selected
                   model on how this is used." OFF)

option(FETCH_ONEDPL "Fetch (download) the oneDPL library for *supported* models. This uses CMake's
                  FetchContent feature. Specify version by setting FETCH_ONEDPL_VERSION" OFF)
set(FETCH_ONEDPL_VERSION "oneDPL-2022.6.0-rc1" CACHE STRING "Specify version of oneTBB to use if FETCH_ONEDPL is ON")

if (FETCH_ONEDPL)
    FetchContent_Declare(
            oneDPL
            GIT_REPOSITORY https://github.com/oneapi-src/oneDPL.git
            GIT_TAG "${FETCH_ONEDPL_VERSION}"
    )
    string(TOLOWER ${USE_ONEDPL} ONEDPL_BACKEND)
    # XXX oneDPL looks for omp instead of openmp, which mismatches(!) with ONEDPL_PAR_BACKEND if using find_package
    if (ONEDPL_BACKEND STREQUAL "openmp")
        set(ONEDPL_BACKEND omp)
    endif ()
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(oneDPL)
    if (NOT oneDPL_POPULATED)
        FetchContent_Populate(oneDPL)
        if (USE_ONETBB)
            macro(find_package NAME)
                if ("${NAME}" STREQUAL "TBB")
                    message(STATUS "Discarding oneDPL's call to find_package(${NAME} ${ARGN})")
                else ()
                    _find_package(${NAME} ${ARGN})
                endif ()
            endmacro()
        endif ()
        add_subdirectory(${onedpl_SOURCE_DIR} ${onedpl_BINARY_DIR} EXCLUDE_FROM_ALL)

        # Fixup oneDPL's omission on setting DPCPP definitions.
        # We do this after the creation of the oneDPL target.
        if (ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
            target_compile_definitions(oneDPL INTERFACE ONEDPL_USE_DPCPP_BACKEND=1)
        endif ()

    endif ()
endif ()


# include our macros
include(cmake/register_models.cmake)

register_flag_optional(ENABLE_MPI "Enables MPI support at compile time, set MPI_HOME (e.g -DMPI_HOME=/usr/lib64/openmpi/) if not on PATH" OFF)
register_flag_optional(ENABLE_PROFILING "Enables kernel profiler, this may introduce synchronisation overhead for some models." OFF)

list(APPEND IMPL_SOURCES
        driver/definitions.cpp
        driver/shared.cpp
        driver/hydro.cpp
        driver/read_input.cpp
        driver/report.cpp
        driver/timer.cpp
        driver/timestep.cpp
        driver/clover_leaf.cpp
        driver/advection.cpp
        driver/update_tile_halo.cpp
        driver/start.cpp
        driver/comms.cpp
        driver/visit.cpp
        driver/mpi_shim.cpp
)

set(MODEL_SRC
        accelerate.cpp
        advec_cell.cpp
        advec_mom.cpp
        build_field.cpp
        calc_dt.cpp
        comms_kernel.cpp
        field_summary.cpp
        flux_calc.cpp
        generate_chunk.cpp
        ideal_gas.cpp
        initialise_chunk.cpp
        initialise.cpp
        finalise.cpp
        pack_kernel.cpp
        PdV.cpp
        reset_field.cpp
        revert.cpp
        update_halo.cpp
        update_tile_halo_kernel.cpp
        viscosity.cpp)

# register our models <model_name> <preprocessor_def_name> <source files...>
register_model(omp USE_OMP ${MODEL_SRC})
register_model(omp-target USE_OMP_TARGET ${MODEL_SRC})
register_model(serial USE_SERIAL ${MODEL_SRC})
register_model(cuda USE_CUDA ${MODEL_SRC})
register_model(hip USE_HIP ${MODEL_SRC})
register_model(std-indices USE_STD ${MODEL_SRC})
register_model(tbb USE_TBB ${MODEL_SRC})
register_model(kokkos USE_KOKKOS ${MODEL_SRC})
register_model(acc ACC ${MODEL_SRC})
register_model(sycl-acc USE_SYCL_ACC ${MODEL_SRC})
register_model(sycl-usm USE_SYCL_USM ${MODEL_SRC})
register_model(raja USE_RAJA ${MODEL_SRC})

set(USAGE ON CACHE BOOL "Whether to print all custom flags for the selected model")

message(STATUS "Available models:  ${REGISTERED_MODELS}")
if (NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL is unspecified, pick one from the available models")
else ()
    string(TOLOWER ${MODEL} MODEL)
    message(STATUS "Selected model  :  ${MODEL}")
endif ()

# load the $MODEL.cmake file and setup the correct IMPL_* based on $MODEL
load_model(${MODEL})

if (USAGE)
    # print the usage of the registered flag options
    registered_flags_action(print RESULT)
    message(STATUS "${RESULT}")
endif ()
# check required/set default for all registered flag options
registered_flags_action(check RESULT)
message(STATUS "${RESULT}")


# run model specific setup, i.e append build flags, etc
setup()

# CMake insists that -O2 (or equivalent) is the universally accepted optimisation level
# we remove that here and use our own <BUILD_TYPE>_FLAGS
if (CMAKE_CXX_FLAGS_${BUILD_TYPE})
    wipe_gcc_style_optimisation_flags(CMAKE_CXX_FLAGS_${BUILD_TYPE})
endif ()

message(STATUS "Default ${CMAKE_BUILD_TYPE} flags are `${DEFAULT_${BUILD_TYPE}_FLAGS}`, set ${BUILD_TYPE}_FLAGS to override (CXX_EXTRA_* flags are not affected)")


# setup common build flag defaults if there are no overrides
if (NOT DEFINED ${BUILD_TYPE}_FLAGS)
    set(ACTUAL_${BUILD_TYPE}_FLAGS ${DEFAULT_${BUILD_TYPE}_FLAGS})
else ()
    set(ACTUAL_${BUILD_TYPE}_FLAGS ${${BUILD_TYPE}_FLAGS})
endif ()

if (NOT ENABLE_MPI)
    list(APPEND IMPL_DEFINITIONS NO_MPI)
else ()
    find_package(MPI REQUIRED)
    list(APPEND LINK_LIBRARIES MPI::MPI_C)
endif ()
if (ENABLE_PROFILING)
    list(APPEND IMPL_DEFINITIONS ENABLE_PROFILING)
endif ()

message(STATUS "CXX vendor  : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "Platform    : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Sources     : ${IMPL_SOURCES}")
message(STATUS "Libraries   : ${LINK_LIBRARIES} ${CXX_EXTRA_LIBRARIES}")

message(STATUS "CXX Flags   : ${CMAKE_CXX_FLAGS_${BUILD_TYPE}} ${ACTUAL_${BUILD_TYPE}_FLAGS} ${CXX_EXTRA_FLAGS}
    CXX flags derived from (CMake + (Override ? Override : Default) + Extras), where:
        CMake                    = `${CMAKE_CXX_FLAGS_${BUILD_TYPE}}`
        Default                  = `${DEFAULT_${BUILD_TYPE}_FLAGS}`
        Override (${BUILD_TYPE}_FLAGS) = `${${BUILD_TYPE}_FLAGS}`
        Extras (CXX_EXTRA_FLAGS) = `${CXX_EXTRA_FLAGS}`")
message(STATUS "CXX Link Flags  : ${LINK_FLAGS} ${CXX_EXTRA_LINK_FLAGS}")
message(STATUS "CXX Linker Flags: ${CMAKE_EXE_LINKER_FLAGS} ${CXX_EXTRA_LINKER_FLAGS} ")
message(STATUS "Defs        : ${IMPL_DEFINITIONS}")
message(STATUS "Executable  : ${BIN_NAME}")

# below we have all the usual CMake target setup steps

include_directories(${CMAKE_BINARY_DIR}/generated)

add_executable(${EXE_NAME} ${IMPL_SOURCES})
target_link_libraries(${EXE_NAME} PUBLIC ${LINK_LIBRARIES} m)
target_compile_definitions(${EXE_NAME} PUBLIC ${IMPL_DEFINITIONS})

if (CXX_EXTRA_LIBRARIES)
    target_link_libraries(${EXE_NAME} PUBLIC ${CXX_EXTRA_LIBRARIES})
endif ()
target_include_directories(${EXE_NAME} PRIVATE driver ${IMPL_DIRECTORIES})

target_compile_options(${EXE_NAME} PUBLIC "$<$<CONFIG:Release>:${ACTUAL_RELEASE_FLAGS};${CXX_EXTRA_FLAGS}>")
target_compile_options(${EXE_NAME} PUBLIC "$<$<CONFIG:Debug>:${ACTUAL_DEBUG_FLAGS};${CXX_EXTRA_FLAGS}>")

target_link_options(${EXE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:LINKER:${CXX_EXTRA_LINKER_FLAGS}>)
target_link_options(${EXE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${LINK_FLAGS};${CXX_EXTRA_LINK_FLAGS}>)

# some models require the target to be already specified so they can finish their setup here
# this only happens if the model.cmake definition contains the `setup_target` macro
if (COMMAND setup_target)
    setup_target(${EXE_NAME})
endif ()

target_compile_definitions(${EXE_NAME} PRIVATE)

#if ((CMAKE_GENERATOR MATCHES "Unix Makefiles") OR (CMAKE_GENERATOR MATCHES "Ninja"))
#    add_custom_target(extract_compile_commands ALL
#            COMMAND ${CMAKE_COMMAND}
#            -DCOMMAND_FILE=${CMAKE_BINARY_DIR}/compile_commands.json
#            -DPRE_CONFIGURE_FILE= "${CMAKE_SOURCE_DIR}/src/meta_build.h.in"
#            -DPOST_CONFIGURE_FILE= "${CMAKE_BINARY_DIR}/generated/meta_build.h"
#            -DSRC_DIR_PREFIX= "${CMAKE_SOURCE_DIR}/src"
#            -DOUT_DIR_PREFIX= "CMakeFiles/${EXE_NAME}.dir"
#            -P ${CMAKE_SOURCE_DIR}/cmake/extract_compile_command.cmake
#            )
#    add_dependencies(${EXE_NAME} extract_compile_commands)
#else ()
#    set(COMPILE_COMMANDS COMPILE_FLAGS)
#endif ()

set_target_properties(${EXE_NAME} PROPERTIES OUTPUT_NAME "${BIN_NAME}")

install(TARGETS ${EXE_NAME} DESTINATION bin)

include(CTest)
if (BUILD_TESTING)
    set(case_timeout 60)
    set(reference_timeout 900)
    if (MODEL STREQUAL "serial")
        set(reference_timeout 3600)
    endif ()
    if (SYCL_COMPILER STREQUAL "ADAPTIVECPP" OR ACPP_PCUDA_DRIVER)
        set(case_timeout 180)
    endif ()

    function(add_solution_case name deck outcome problem)
        cmake_parse_arguments(CASE "" "INVARIANTS;TIMEOUT" "OUTPUT;REPORT" ${ARGN})
        if (NOT CASE_INVARIANTS)
            set(CASE_INVARIANTS PASSED)
        endif ()
        if (NOT CASE_TIMEOUT)
            set(CASE_TIMEOUT ${case_timeout})
        endif ()
        set(ranks 1)
        if (ENABLE_MPI AND MPIEXEC_EXECUTABLE)
            list(APPEND ranks 2)
        endif ()
        foreach(rank_count IN LISTS ranks)
            set(test "solution_${name}")
            set(threads 2)
            if (rank_count EQUAL 2)
                string(APPEND test "_mpi")
                set(threads 1)
            endif ()
            set(directory "${CMAKE_CURRENT_BINARY_DIR}/tests/${test}")
            file(MAKE_DIRECTORY "${directory}")
            set(input "${directory}/clover.in")
            set(report "${directory}/clover.out")
            file(WRITE "${input}" "${deck}")

            set(expected_exit 0)
            if (outcome STREQUAL "FAILED")
                set(expected_exit 1)
            elseif (outcome STREQUAL "ABORT")
                set(expected_exit nonzero)
            endif ()
            set(command "$<TARGET_FILE:${EXE_NAME}>" --file "${input}" --out "${report}")
            if (rank_count EQUAL 2)
                set(command "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    sh ${MPIEXEC_POSTFLAGS} "${CMAKE_CURRENT_SOURCE_DIR}/tests/mpi_rank_check.sh"
                    "$<TARGET_FILE:${EXE_NAME}>" "${input}" "${report}")
                if (NOT outcome STREQUAL "ABORT")
                    list(APPEND command "${expected_exit}")
                    set(expected_exit 0)
                endif ()
            endif ()

            set(expected_output " - Total ranks: ${rank_count}")
            if (rank_count EQUAL 2 AND outcome STREQUAL "ABORT")
                list(APPEND expected_output " Launching hydro")
            else ()
                list(APPEND expected_output ${CASE_OUTPUT})
            endif ()
            if (NOT outcome STREQUAL "ABORT")
                list(APPEND expected_output " - Outcome: ${outcome}" " - Problem: ${problem}"
                     " - Invariants: ${CASE_INVARIANTS}")
            endif ()
            add_test(NAME ${test}
                COMMAND ${CMAKE_COMMAND} "-DTEST_COMMAND=${command}" "-DEXPECT_EXIT=${expected_exit}"
                        "-DEXPECT_OUTPUT=${expected_output}" "-DEXPECT_REPORT=${CASE_REPORT}"
                        "-DFORBID_PASSED=$<STREQUAL:${outcome},SKIPPED>" "-DREPORT=${report}"
                        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check.cmake")
            set_tests_properties(${test} PROPERTIES
                WORKING_DIRECTORY "${directory}" TIMEOUT ${CASE_TIMEOUT} PROCESSORS 2
                ENVIRONMENT "OMP_NUM_THREADS=${threads};ACPP_APPDB_DIR=${directory}/jit")
        endforeach()
    endfunction()

    set(reference_input "${PROJECT_SOURCE_DIR}/InputDecks/clover_bm16_short.in")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${reference_input}")
    file(READ "${reference_input}" reference_deck)
    add_solution_case(reference "${reference_deck}" PASSED 4 TIMEOUT ${reference_timeout})

    set(base_deck "*clover
state 1 density=0.2 energy=1.0
state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0
x_cells=10
y_cells=2
xmin=0.0
ymin=0.0
xmax=10.0
ymax=2.0
initial_timestep=0.04
timestep_rise=1.5
max_timestep=0.04
end_time=3.0
")
    string(REPLACE "energy=2.5" "energy=2.0" wrong_deck "${base_deck}")
    add_solution_case(wrong_solution "${wrong_deck}test_problem 1\n" FAILED 1)
    add_solution_case(unchecked "${base_deck}*endclover\n" SKIPPED none)
    add_solution_case(disabled "${base_deck}test_problem 0\n" SKIPPED none)
    add_solution_case(unknown "${base_deck}test_problem 999" FAILED 999)
    add_solution_case(negative "${base_deck}test_problem -1\n" FAILED -1)
    string(REPLACE " " "\t" tab_deck "${base_deck}")
    add_solution_case(whitespace " \t\n${tab_deck}\t test_problem\t=\t1\r" PASSED 1)
    add_solution_case(arbitrary_stop "${base_deck}end_step=7\nsummary_frequency=0\n" SKIPPED none)
    string(REPLACE "x_cells=10" "x_cells=23" different_deck "${base_deck}")
    string(REPLACE "y_cells=2" "y_cells=11" different_deck "${different_deck}")
    add_solution_case(different_mesh "${different_deck}end_step=13\nsummary_frequency=3\n" SKIPPED none)
    string(REGEX REPLACE "density=[0-9.]+" "density=0.0" zero_mass_deck "${base_deck}")
    add_solution_case(zero_mass "${zero_mass_deck}end_step=1\n" FAILED none INVARIANTS FAILED)

    set(cell_deck "*clover
state 1 density=1.0 energy=2.5
x_cells=20
y_cells=12
xmin=0.0
ymin=0.0
xmax=20.0
ymax=12.0
initial_timestep=0.000001
max_timestep=0.000001
end_step=1
summary_frequency=0
tiles_per_chunk=2
")
    foreach(field IN ITEMS density energy)
        set(state "density=1.0 energy=2.5")
        string(REGEX REPLACE "${field}=[0-9.]+" "${field}=-0.1" state "${state}")
        add_solution_case(negative_cell_${field}
            "${cell_deck}state 2 ${state} geometry=rectangle xmin=15.0 xmax=17.0 ymin=7.0 ymax=9.0\n"
            FAILED none INVARIANTS FAILED OUTPUT "FAILED at step 0: finite summaries=1, invalid cells=4")
    endforeach()
    add_solution_case(nonfinite_cell
        "${cell_deck}state 2 density=1.0 energy=nan geometry=rectangle xmin=15.0 xmax=17.0 ymin=7.0 ymax=9.0\n"
        FAILED none INVARIANTS FAILED OUTPUT "FAILED at step 0: finite summaries=0, invalid cells=4")

    string(REPLACE "energy=2.5" "energy=0.0" cold_deck "${cell_deck}")
    add_solution_case(cold_equilibrium "${cold_deck}end_step=7\n" SKIPPED none
        OUTPUT "Invalid cells at final summary: 0" "Total energy change: 0")
    add_solution_case(invalid_timestep "${cell_deck}max_timestep=0.0\n" ABORT none
        OUTPUT "invalid timestep or time does not advance")
    # A positive dt above dtmin, but below one ULP of the accumulated time.
    add_solution_case(stalled_time
        "${cold_deck}initial_timestep=1.0e30\nmax_timestep=1.0e30\ntimestep_rise=1.0e-10\nend_time=1.0e40\nend_step=5\n"
        ABORT none OUTPUT "invalid timestep or time does not advance")

    # UK-MAC Serial b9a2b9c / OpenMP 022eb96 / ref 0fdb917, Fortran kernels: negative fluxes across unequal tiles
    set(circle_deck "*clover
state 1 density=0.2 energy=1.0
state 2 density=1.0 energy=2.5 geometry=circle xmin=4.0 ymin=1.0 radius=0.6
x_cells=23
y_cells=11
xmin=0.0
ymin=0.0
xmax=10.0
ymax=2.0
initial_timestep=0.004
max_timestep=0.04
end_time=3.0
end_step=30
summary_frequency=1
")
    foreach(tiles IN ITEMS 1 6)
        add_solution_case(circle_${tiles}_tiles "${circle_deck}tiles_per_chunk=${tiles}\n" SKIPPED none
            REPORT "   1.256649e-01   6.283245e+00   3.573257e-01")
    endforeach()
endif ()
