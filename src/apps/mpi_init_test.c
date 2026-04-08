#include "info_op.h"
#include "mpi.h"

int main(int argc, char *argv[]){
    int foo_before_mpi_init = 123;
    int num_processor, myrank;
    num_processor = 4;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processor);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int foo_after_mpi_init = 456;
    if (myrank == 0) {
      foo_after_mpi_init = 789;
      /* debug_proc(myrank, "changed var foo = %d\n", foo_after_mpi_init); */
      debug_proc(myrank, "%d procs in total\n", num_processor);
    }
    debug_proc(myrank, "I get var foo = %d\n", foo_before_mpi_init);
    MPI_Finalize();
}
