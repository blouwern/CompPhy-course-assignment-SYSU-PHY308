cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(MPIAssignment)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(APP_DIR ${SOURCE_DIR}/apps)
set(COMMON_DIR ${SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(INFO_OP ${COMMON_DIR}/info_op.c)
set(INFO_OP_OPENMP ${COMMON_DIR}/info_op_openmp.c)
set(MATRIX_OP_BASE ${COMMON_DIR}/matrix_op_base.c)
set(MATRIX_OP_CBLAS ${COMMON_DIR}/matrix_op_cblas.c)

add_executable(comprehensive_demo
  ${APP_DIR}/comprehensive_demo.c
  ${INFO_OP}
  ${MATRIX_OP_BASE}
  ${MATRIX_OP_CBLAS})
target_link_libraries(comprehensive_demo MPI::MPI_C
  BLAS::BLAS
  m)
target_compile_definitions(comprehensive_demo PRIVATE NON_BLOCKING_COMMUNICATION=1)
target_compile_definitions(comprehensive_demo PRIVATE MPI_COMPUTATION_USE_BLAS=1)
target_compile_definitions(comprehensive_demo PRIVATE ADD_RSSEQ_COMPARE=1)
target_compile_definitions(comprehensive_demo PRIVATE REPORT_DEBUG_INFO=1)

add_executable(MM_cblas_mpi
  ${APP_DIR}/MM_cblas_mpi.c
  ${INFO_OP}
  ${MATRIX_OP_BASE}
  ${MATRIX_OP_CBLAS})
target_link_libraries(MM_cblas_mpi MPI::MPI_C
  BLAS::BLAS
  m)
target_compile_definitions(MM_cblas_mpi PRIVATE NON_BLOCKING_COMMUNICATION=1)
target_compile_definitions(MM_cblas_mpi PRIVATE REPORT_DEBUG_INFO=0)

add_executable(MM_RS_mpi
  ${APP_DIR}/MM_RS_mpi.c
  ${INFO_OP}
  ${MATRIX_OP_BASE})
target_link_libraries(MM_RS_mpi MPI::MPI_C
  BLAS::BLAS
  m)
target_compile_definitions(MM_RS_mpi PRIVATE NON_BLOCKING_COMMUNICATION=1)
target_compile_definitions(MM_RS_mpi PRIVATE REPORT_DEBUG_INFO=0)

add_executable(MM_cblas_seq
  ${APP_DIR}/MM_cblas_seq.c
  ${INFO_OP}
  ${MATRIX_OP_BASE})
target_link_libraries(MM_cblas_seq BLAS::BLAS
  m)

add_executable(MM_RS_seq
  ${APP_DIR}/MM_RS_seq.c
  ${INFO_OP}
  ${MATRIX_OP_BASE})
target_link_libraries(MM_RS_seq
  m)

add_executable(MM_cblas_openmp
  ${APP_DIR}/MM_cblas_openmp.c
  ${INFO_OP_OPENMP}
  ${INFO_OP}
  ${MATRIX_OP_BASE}
  ${MATRIX_OP_CBLAS})
target_link_libraries(MM_cblas_openmp BLAS::BLAS
  m)
if(OpenMP_C_FOUND)
  target_link_libraries(MM_cblas_openmp OpenMP::OpenMP_C)
endif()
target_compile_definitions(MM_cblas_openmp PRIVATE REPORT_DEBUG_INFO=1)

add_executable(MM_RS_openmp
  ${APP_DIR}/MM_RS_openmp.c
  ${INFO_OP_OPENMP}
  ${INFO_OP}
  ${MATRIX_OP_BASE})
target_link_libraries(MM_RS_openmp BLAS::BLAS
  m)
if(OpenMP_C_FOUND)
  target_link_libraries(MM_RS_openmp OpenMP::OpenMP_C)
endif()
target_compile_definitions(MM_RS_openmp PRIVATE REPORT_DEBUG_INFO=1)

