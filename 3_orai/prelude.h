/**
 * = CUDA error checking utility =
 * Created by Bálint Tóth
 * University of Pannonia,
 * Bioelectric Brain Imaging Laboratory
 * 2025
 * 
 * This is a single header containing 3 macros that can be used to wrap cuda function calls and
 * automatically divert the control flow if any error occures.
 * 
 * Available macros:
 *  - CUDA_ERROR_CHECK(CALL): Check the result of the wrapped function call and abort the program if it returned an error.
 *    - CALL: Expected to be a function call returning `cudaError_t`.
 *  - CUDA_GOTO_ON_ERROR(CALL, RETVAR, GOTO_TAG, LOG_TAG, LOGFMT, ...):
 *    - CALL: Expected to be a function call returning `cudaError_t`.
 *    - RETVAR: Name of the `cudaError_t` type status variable to be set to the error result type.
 *    - GOTO_TAG: Tag in the context of the calling function to jump out to.
 *    - LOG_TAG: Label for the log message display.  Can be defined label or any char* variable.
 *    - LOGFMT: Error message format in printf format syntax.
 *    - ...: Additional variadic arguments to printf.
 *  - CUDA_GOTO_ON_FALSE(EXPR, RETVAR, RETVAL, GOTO_TAG, LOG_TAG, LOGFMT, ...):
 *    - CALL: Expected to be a function call returning `cudaError_t`.
 *    - RETVAR: Name of the `cudaError_t` type status variable to be set to the error result type.
 *    - RETVAL: `cudaError_t` type value of the status variable.
 *    - GOTO_TAG: Tag in the context of the calling function to jump out to.
 *    - LOG_TAG: Label for the log message display.  Can be defined label or any char* variable.
 *    - LOGFMT: Error message format in printf format syntax.
 *    - ...: Additional variadic arguments to printf.
 *  - CUDA_LASTERR(): Error check barrier. Queries the last error from CUDA and aborts the program if the returned value is an error code.
 * 
 * To use: simply include the header. To configure error handling behaviour, before the `#include` directive
 * the following config flags can be defined:
 *  - CUDA_ERROR_CHECK_NO_ABORT: Display error message and description and continue program execution.
 *  - CUDA_ERROR_CHECK_SILENT: Turn off error checking, only execute the wrapped call.
 *
 * 
 * Usage example:
 * 
 * main.cpp
 * ```
 *  #include <cuda.h>
 *  
 *  // make it so that an error check do not abort the program
 *  #define CUDA_ERROR_CHECK_NO_ABORT
 *  #include "cuda_error.h"
 *  
 *  static const char* TAG = "main";
 *  
 *  cudaError_t kernel_driver(int a)
 *  {
 * 	 int* d_a;
 *    cudaError_t ret = cudaSuccess;
 *    // attempt the allocation and copy operations
 *    CUDA_GOTO_ON_ERROR(cudaMalloc(&d_a, sizeof(int)), ret, panic, TAG, "Malloc failed");
 *    CUDA_GOTO_ON_ERROR(cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice), ret, memcpy_panic, TAG, "Memcpy failed :(");
 * 
 *    // if copy was successful, launch a kernel
 *    kernel<<<1,1>>>(d_a);
 *	  CUDA_LASTERR();
 *    
 *    // attempt to free the variable
 *    CUDA_GOTO_ON_ERROR(cudaFree(d_a), ret, panic, TAG, "Free failed");
 *  
 *    return cudaSuccess;
 *  memcpy_panic:
 *    cudaFree(d_a);
 *  panic:
 *    return ret;
 *  }
 * 
 *  int main()
 *  {
 *    CUDA_ERROR_CHECK(kernel_driver(69));
 *    return 0;
 *  }
 * ```
 *
 * Discalimer: The C preprocessor is not checking the types of macro parameters, 
 * so careful consideration is required while using this header. If any parameter
 * passed to these macros contains commas ( ',' ), the preprocessor will substitute 
 * the rest of the expression as a different parameter, so it is advised to wrap every
 * non-trivial parameter in parenthesis, otherwise hard to trace compile time errors
 * can occur.
 * For this reason some consider function-like C macros as antipattern.
*/

#ifndef PRELUDE_H
#define PRELUDE_H

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#if defined (ERROR_CHECK_NO_ABORT)
#  define CUDA_ABORT_CALL ((void)0)
#else
#  define CUDA_ABORT_CALL abort()
#endif

#if defined (ERROR_CHECK_SILENT)

#define CUDA_ERROR_CHECK(CALL) (CALL)
#define CUDA_GOTO_ON_ERROR(CALL, RETVAR, GOTO_TAG, LOG_TAG, LOGFMT, ...) (CALL)
#define CUDA_GOTO_ON_FALSE(EXPR, RETVAR, RETVAL, GOTO_TAG, LOG_TAG, LOGFMT, ...) (EXPR)
#define CUDA_LASTERR() ((void)0)
#define CUDA_THROW_ON_ERROR(CALL) (CALL)

#else

#define CUDA_ERROR_CHECK(CALL)                                       \
do {                                                                 \
    cudaError_t res = cudaSuccess;                                   \
    if ((res = (CALL)) != cudaSuccess) {                             \
        fprintf(stderr, "[E] %s:%d:%s returned error \"%s\" (%s)\n"  \
            "\nABORTING\n\n",                                        \
            __FILE__,                                                \
            __LINE__,                                                \
            #CALL,                                                   \
            cudaGetErrorName(res),                                   \
            cudaGetErrorString(res));                                \
            CUDA_ABORT_CALL;                                         \
    }                                                                \
} while (0)

#define CUDA_GOTO_ON_ERROR(CALL, RETVAR, GOTO_TAG, LOG_TAG, LOGFMT, ...) \
do {                                                                     \
    if ((RETVAR = (CALL)) != cudaSuccess) {                              \
        fprintf(stderr, "[E] %s:%d:%s returned error \"%s\" (%s)\n"      \
            "    %s - " LOGFMT "\n",                                     \
            __FILE__,                                                    \
            __LINE__,                                                    \
            #CALL,                                                       \
            cudaGetErrorName(RETVAR),                                    \
            cudaGetErrorString(RETVAR),                                  \
            LOG_TAG,                                                     \
			##__VA_ARGS__);                                              \
        goto GOTO_TAG;                                                   \
    }                                                                    \
} while (0)

#define CUDA_GOTO_ON_FALSE(EXPR, RETVAR, RETVAL, GOTO_TAG, LOG_TAG, LOGFMT, ...) \
do {                                                                             \
    if (!(EXPR)) {                                                               \
        RETVAR = (RETVAL);                                                       \
        fprintf(stderr, "[E] %s:%d:%s evaluated false\n"                         \
            "    %s - " LOGFMT "\n",                                             \
            __FILE__,                                                            \
            __LINE__,                                                            \
            #EXPR,                                                               \
            LOG_TAG,                                                             \
			##__VA_ARGS__);                                                      \
        goto GOTO_TAG;                                                           \
    }                                                                            \
} while (0)

#define CUDA_LASTERR()                                                       \
do {                                                                         \
	cudaError_t res = cudaGetLastError();                                    \
	if (res != cudaSuccess) {                                                \
		fprintf(stderr, "[E] %s:%d Check barrier found error \"%s\" (%s)\n"  \
			"\nABORTING\n\n",                                                \
			__FILE__,                                                        \
			__LINE__,                                                        \
			cudaGetErrorName(res),                                           \
			cudaGetErrorString(res));                                        \
			CUDA_ABORT_CALL;                                                 \
	}                                                                        \
} while (0)
	
#endif


#define CUDA_TIME_START()                         \
	static cudaEvent_t cuda_measure_start;        \
	static cudaEvent_t cuda_measure_end;          \
	static float cuda_measure_elapsed;            \
    cudaEventCreate(&cuda_measure_start);         \
    cudaEventCreate(&cuda_measure_end);           \
    printf("[TIME] CUDA measurement started.\n"); \
    cudaEventRecord(cuda_measure_start, 0)

#define CUDA_TIME_END()                                                                \
    cudaEventRecord(cuda_measure_end, 0);                                              \
    printf("[TIME] CUDA measurement ended.\n");                                        \
    cudaEventSynchronize(cuda_measure_end);                                            \
    cudaEventElapsedTime(&cuda_measure_elapsed, cuda_measure_start, cuda_measure_end); \
    cudaEventDestroy(cuda_measure_start);                                              \
    cudaEventDestroy(cuda_measure_end);                                                \
    printf("[TIME] CUDA elapsed time = %f ms\n", cuda_measure_elapsed)

#endif // PRELUDE_H