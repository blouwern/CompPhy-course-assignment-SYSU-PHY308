#include "matrix_op.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double** make_matrix(int n_row, int n_col) {
    double** M = (double**)malloc(n_row * sizeof(double*));
    M[0] = (double*)malloc(n_row * n_col * sizeof(double));
    for (int i = 1; i < n_row; i++) {
        M[i] = M[0] + i * n_col;
    }
    return M;
}

double** generate_matrix(int n_row, int n_col) {
    double** M = make_matrix(n_row, n_col);
    srand(time(NULL) + n_row);

    double scale_factor = 1.0 / sqrt((double)n_col);

    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            double rand_val = (double)random() / RAND_MAX;
            rand_val = (rand_val * 2.0) - 1.0;
            M[i][j] = rand_val * scale_factor;
        }
    }
    return M;
}

double** transform_matrix(double** M, int n_row, int n_col) {
    double** M_T = (double**)malloc(n_col * sizeof(double*));
    M_T[0] = (double*)malloc(n_row * n_col * sizeof(double));
    for (int i = 1; i < n_col; i++) {
        M_T[i] = M_T[0] + i * n_row;
    }
    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            M_T[j][i] = M[i][j];
        }
    }
    return M_T;
}

double* linearlize_matrix(double** M, int n_row, int n_col) {
    double* L_M = (double*)malloc(n_row * n_col * sizeof(double));
    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            L_M[i * n_col + j] = M[i][j];
        }
    }
    return L_M;
}

double* matrix_multiply_matrix_linear(double* A_L, double* B_L, int n_row_A,
                                      int n_col_A, int n_col_B) {
    double* result = (double*)malloc(n_row_A * n_col_B * sizeof(double));
    if (result == NULL) {
        return NULL;
    }

    for (int i = 0; i < n_row_A; i++) {
        for (int j = 0; j < n_col_B; j++) {
            result[i * n_col_B + j] = 0.0;
            for (int k = 0; k < n_col_A; k++) {
                result[i * n_col_B + j] +=
                    A_L[i * n_col_A + k] * B_L[k * n_col_B + j];
            }
        }
    }
    return result;
}

void check_matrix_equal(double** M1, double** M2, int n_row, int n_col) {
    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            double diff = M1[i][j] - M2[i][j];
            if (diff > 1e-6 || diff < -1e-6) {
                printf("Matrices are not equal!\n Element %d row %d column.", i, j);
                return;
            }
        }
    }
    printf("Two matrix are equal.\n");
}
