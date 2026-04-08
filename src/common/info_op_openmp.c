#include "info_op.h"

#include <stdarg.h>
#if REPORT_DEBUG_INFO
#include <stdio.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

void debug_thread(int thread_id, const char* message, ...) {
#if REPORT_DEBUG_INFO
    va_list args;
    va_start(args, message);
#ifdef _OPENMP
#pragma omp critical(debug_thread_log)
#endif
    {
        printf("|%12.6f|T%2d|", get_timer(), thread_id);
        vprintf(message, args);
        fflush(stdout);
    }
    va_end(args);
#else
    (void)thread_id;
    (void)message;
#endif
}
