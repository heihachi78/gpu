#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "prelude.h"
#include <chrono>

void mxm_serial(int N, float* a, float* b, float* c)
{
    // every item of the result
    for(int i=0; i<N; i++)
    {
        // is equal to the sum of every row
        for(int j=0; j<N; j++)
        {
            // multiplied by every column
            float tmp = 0.0f;
            for(int k=0; k<N; k++)
            {
                tmp += a[i*N+k] * b[k*N+j];
            }
            c[i*N+j] = tmp;
        }
    }
}

void const_init(int N, float* mat, float init)
{
    // init every element to init value
    for(int i=0; i<N*N; i++)
    {
        mat[i] = init;
    }
}

void diag_init(int N, float* mat, float init)
{
    // init every element to 0
    for(int i=0; i<N*N; i++)
    {
        mat[i] = 0.0;
    }
    
    // init the diagonal to init value
    for(int i=0; i<N; i++)
    {
        mat[i*N+i] = init;
    }
}

void mxm_test_serial(int N)
{
    // heap allocate memory
    float* a = new float[N*N];
    float* b = new float[N*N];
    float* c = new float[N*N];

    // init matrices
    const_init(N, a, 2.5);
    const_init(N, c, 0.0);
    diag_init(N, b, 1.0);

    // call our function and measure it
    auto start = std::chrono::high_resolution_clock::now();
    mxm_serial(N, a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // printing the result
    printf("matrix size: %d, elapsed time: %f\n", N, float(elapsed_us)/1e3);

    // print the result matrix
    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            // printf("%0.2f ", c[i*N+j]);
        }
        // printf("\n");
    }

    // free memory
    delete[] a, b, c;
}

int main()
{
    mxm_test_serial(32);
    return 0;
}