#pragma once

double** make_matrix(int n_row, int n_col);
double** generate_matrix(int n_row, int n_col);
double** transform_matrix(double** M, int n_row, int n_col);
double* linearlize_matrix(double** M, int n_row, int n_col);
double* matrix_multiply_matrix_linear(double* A_L, double* B_L, int n_row_A,
                                      int n_col_A, int n_col_B);
double* matrix_multiply_matrix_linear_cblas(double* A_L, double* B_L,
                                            int n_row_A, int n_col_A,
                                            int n_col_B);
void check_matrix_equal(double** M1, double** M2, int n_row, int n_col);
