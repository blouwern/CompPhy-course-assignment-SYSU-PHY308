#include "cblas.h"
#include "info_op.h"
#include "matrix_op.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef _OPENMP
#    include "omp.h"
#else
static int omp_get_max_threads(void) {
    return 1;
}

static int omp_get_thread_num(void) {
    return 0;
}
#endif

int main(int argc, char* argv[]) {
    const char* module_name = "CBLAS OpenMP";

    int user_threads = 0;
    int opt;
    while ((opt = getopt(argc, argv, "t:")) != -1) {
        switch (opt) {
        case 't':
            user_threads = atoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s [-t N]\n", argv[0]);
            return 1;
        }
    }
    int n_threads;
    if (user_threads > 0) {
        n_threads = user_threads;
    } else {
        n_threads = omp_get_num_procs(); // 获取硬件核心数
        printf("No -t provided, using all available cores: %d\n", n_threads);
    }
    omp_set_num_threads(n_threads);
    omp_set_dynamic(0);

    int n_matrix_A_row, n_matrix_A_col, n_matrix_B_row, n_matrix_B_col;
    if (argc == 1) {
        n_matrix_A_row = 1000;
        n_matrix_A_col = 500;
        n_matrix_B_row = n_matrix_A_col;
        n_matrix_B_col = 2000;
    } else if (argc == 5) {
        n_matrix_A_row = atoi(argv[1]);
        n_matrix_A_col = atoi(argv[2]);
        n_matrix_B_row = atoi(argv[3]);
        n_matrix_B_col = atoi(argv[4]);
        if (n_matrix_A_col != n_matrix_B_row) {
            fprintf(stderr,
                    "Error: n_matrix_A_col should be equal to n_matrix_B_row\n");
            return 1;
        }
    } else {
        fprintf(stderr,
                "Usage: %s [n_matrix_A_row n_matrix_A_col n_matrix_B_row "
                "n_matrix_B_col]\n",
                argv[0]);
        return 1;
    }

    double** matrix_A = generate_matrix(n_matrix_A_row, n_matrix_A_col);
    double** matrix_B = generate_matrix(n_matrix_B_row, n_matrix_B_col);
    double** matrix_result = make_matrix(n_matrix_A_row, n_matrix_B_col);
    double** matrix_result_cblas = make_matrix(n_matrix_A_row, n_matrix_B_col);

    double* L_A = linearlize_matrix(matrix_A, n_matrix_A_row, n_matrix_A_col);
    double* L_B = linearlize_matrix(matrix_B, n_matrix_B_row, n_matrix_B_col);

    int num_thread = omp_get_max_threads();
    if (num_thread < 1) {
        num_thread = 1;
    }

    int n_avg_task_row = n_matrix_A_row / num_thread;
    int n_remainder_task_row = n_matrix_A_row % num_thread;

    double swtime = get_timer();

#pragma omp parallel num_threads(num_thread)
    {
        int thread_id = omp_get_thread_num();
        int n_row_work = thread_id < n_remainder_task_row ? n_avg_task_row + 1 : n_avg_task_row;
        int row_offset = thread_id < n_remainder_task_row ? thread_id * (n_avg_task_row + 1) : n_remainder_task_row * (n_avg_task_row + 1) + (thread_id - n_remainder_task_row) * n_avg_task_row;

        if (n_row_work > 0) {
            debug_thread(thread_id, "Computing %d rows from offset %d\n",
                         n_row_work, row_offset);
            double* L_result_local = matrix_multiply_matrix_linear_cblas(
                L_A + row_offset * n_matrix_A_col, L_B, n_row_work, n_matrix_A_col,
                n_matrix_B_col);

            for (int i = 0; i < n_row_work; i++) {
                for (int j = 0; j < n_matrix_B_col; j++) {
                    matrix_result[row_offset + i][j] =
                        L_result_local[i * n_matrix_B_col + j];
                }
            }
            free(L_result_local);
            debug_thread(thread_id, "Finished %d rows\n", n_row_work);
        }
    }

    double ewtime = get_timer();
    printf("Computation completed.\n");
    printf("Result matrix showcase:\n");
    print_matrix_less(matrix_result, n_matrix_A_row, n_matrix_B_col, 5, 5);
    printf("[Time taken]<%s> : %f seconds\n", module_name, ewtime - swtime);

    printf("Checking result from OpenMP by CBLAS result:\n");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_matrix_A_row,
                n_matrix_B_col, n_matrix_A_col, 1.0, matrix_A[0], n_matrix_A_col,
                matrix_B[0], n_matrix_B_col, 0.0, matrix_result_cblas[0],
                n_matrix_B_col);
    check_matrix_equal(matrix_result, matrix_result_cblas, n_matrix_A_row,
                       n_matrix_B_col);

    free(matrix_A[0]);
    free(matrix_A);
    free(matrix_B[0]);
    free(matrix_B);
    free(matrix_result[0]);
    free(matrix_result);
    free(matrix_result_cblas[0]);
    free(matrix_result_cblas);
    free(L_A);
    free(L_B);

    return 0;
}
