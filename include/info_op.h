#pragma once

double get_timer(void);
void debug_proc(int rank, const char* message, ...);
void debug_thread(int thread_id, const char* message, ...);
void print_matrix(double** matrix, int n_row, int n_col);
void print_matrix_less(double** matrix, int n_row, int n_col, int n_show_row,
                       int n_show_col);
void print_L_matrix_less(double* matrix_L, int n_row, int n_col, int n_show_row,
                         int n_show_col);
