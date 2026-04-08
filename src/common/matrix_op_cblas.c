#include "matrix_op.h"

#include "cblas.h"

#include <stdlib.h>

double* matrix_multiply_matrix_linear_cblas(double* A_L, double* B_L,
                                            int n_row_A, int n_col_A,
                                            int n_col_B) {
    double* result = (double*)malloc(n_row_A * n_col_B * sizeof(double));
    if (result == NULL) {
        return NULL;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_row_A, n_col_B,
                n_col_A, 1.0, A_L, n_col_A, B_L, n_col_B, 0.0, result,
                n_col_B);

    return result;
}
