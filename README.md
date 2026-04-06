# Assignment 1: MPI matrix multiplication.
prequisites: cmake, ninja, openmpi, openblas (`sudo apt install openmpi libopenblas-dev`)
```bash
mkdir build
cd build
cmake -G Ninja ..
ninja
mpirun ./MPIMatrixMultiply_bsp [OPTIONAL:<matrix dimension of A & B>]
```
by default result = A(1000, 500) * B(500, 2000).
