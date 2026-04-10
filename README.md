# MPI/OpenMP Matrix Multiplication Assignment

## Table of Contents
- [MPI/OpenMP Matrix Multiplication Assignment](#mpiopenmp-matrix-multiplication-assignment)
  - [Table of Contents](#table-of-contents)
  - [Repository Overview](#repository-overview)
  - [Prerequisites](#prerequisites)
  - [Build](#build)
  - [Quick Start](#quick-start)
  - [Program Usage](#program-usage)
    - [MPI Programs](#mpi-programs)
    - [OpenMP Programs](#openmp-programs)
    - [Sequential Programs](#sequential-programs)
  - [Benchmark Scripts](#benchmark-scripts)
    - [1) Algorithm-vs-size benchmark](#1-algorithm-vs-size-benchmark)
    - [2) Node/thread scaling benchmark](#2-nodethread-scaling-benchmark)
  - [Report](#report)

## Repository Overview
This repository contains implementations of matrix-matrix multiplication with:

1. MPI parallelism
2. OpenMP parallelism
3. Sequential execution

Each parallel method is implemented with two compute kernels:

1. Rough Simple (manual triple-loop multiplication)
2. CBLAS (`cblas_dgemm`)

The project also includes Python benchmarking scripts and an experiment report.

## Prerequisites
Install the following tools and libraries:

1. CMake
2. Ninja (recommended)
3. MPI implementation (OpenMPI)
4. BLAS/OpenBLAS
5. Python 3 (for benchmark scripts)

Example on Debian/Ubuntu:

```bash
sudo apt install cmake ninja-build openmpi-bin libopenmpi-dev libopenblas-dev python3
```

## Build
From the repository root:

```bash
mkdir -p build
cd build
cmake -G Ninja ..
ninja
```

## Quick Start
Run the following commands from the repository root to build and run a minimal reproducible test:

```bash
# 1) Build
mkdir -p build
cd build
cmake -G Ninja ..
ninja

# 2) Run one MPI and one OpenMP sample
mpirun -np 4 ./MM_RS_mpi 1000 1000 1000 1000
./MM_RS_openmp -t 4 1000 1000 1000 1000

# 3) Back to root and run smoke benchmarks
cd ..
python3 scripts/performance_test.py --runs 1 --n-values 10 --out report/data/perf_results_6_programs_smoke.csv
python3 scripts/node_scaling_test.py --programs MM_RS_mpi MM_RS_openmp --mpi-counts 2 --omp-threads 1 --n 10 --runs 1 --out report/data/perf_results_node_scaling_smoke.csv
```

If all commands finish successfully, your environment is ready for full-scale tests.

This generates the following executables:

1. `MM_cblas_mpi`
2. `MM_cblas_openmp`
3. `MM_cblas_seq`
4. `MM_RS_mpi`
5. `MM_RS_openmp`
6. `MM_RS_seq`
7. `comprehensive_demo`

## Program Usage
All matrix programs accept optional dimensions:

```text
[n_matrix_A_row n_matrix_A_col n_matrix_B_row n_matrix_B_col]
```

If omitted, each program uses its internal default dimensions.

### MPI Programs
Run MPI targets with `mpirun`:

```bash
cd build
mpirun -np 8 ./MM_RS_mpi 2000 2000 2000 2000
mpirun -np 8 ./MM_cblas_mpi 2000 2000 2000 2000
```

### OpenMP Programs
OpenMP targets support thread count via `-t`:

```bash
cd build
./MM_RS_openmp -t 8 2000 2000 2000 2000
./MM_cblas_openmp -t 8 2000 2000 2000 2000
```

### Sequential Programs

```bash
cd build
./MM_RS_seq 2000 2000 2000 2000
./MM_cblas_seq 2000 2000 2000 2000
```

## Benchmark Scripts
Scripts are located in `scripts`.

### 1) Algorithm-vs-size benchmark
Script: `scripts/performance_test.py`

Purpose:
benchmark the six executables across multiple matrix sizes.

Default output:
`report/data/perf_results_6_programs.csv`

Example:

```bash
python3 scripts/performance_test.py
```

### 2) Node/thread scaling benchmark
Script: `scripts/node_scaling_test.py`

Purpose:
benchmark MPI/OpenMP scaling vs process/thread count.

Default output:
`report/data/perf_results_node_scaling.csv`

Example:

```bash
python3 scripts/node_scaling_test.py
```

## Report
Main report file:

`report/assignment-report-1.md`

Report assets are organized as:

1. `report/data`: CSV benchmark results
2. `report/figures/perf`: algorithm-vs-size plots
3. `report/figures/scaling`: node/thread scaling plots
4. `report/figures/monitoring`: CPU monitoring screenshots
