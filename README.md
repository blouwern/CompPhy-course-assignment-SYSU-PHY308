# List of Contents
[Assignment 1: MPI parallel matrix multiplication](#assignment-1-mpi-parallel-matrix-multiplication)

## Assignment 1: MPI parallel matrix multiplication
Prerequisites: cmake, ninja, openmpi, openblas... Just add what you lack. (e.g. `sudo apt install openmpi libopenblas-dev`)
```bash
mkdir build
cd build
cmake -G Ninja ..
ninja
mpirun ./MPIMatrixMultiply_bsp [OPTIONAL:<matrix dimension of A & B>]
```
by default result = A(1000, 500) * B(500, 2000).
