#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "prelude.h"
#include <chrono>

__global__ void mxm_naive_kernel(int N, float* a, float* b, float* c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(i>=N || j>=N)
    {
        return;
    }

    float tmp = 0;
    for (int k=0; k<N; k++){
        tmp += a[i*N+k] * b[k*N+j];
    }
    c[i * N + j] = tmp;
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

void mxm_test_gpu(int N)
{
    // heap allocate memory
    float* a = new float[N*N];
    float* b = new float[N*N];
    float* c = new float[N*N];

    // init matrices
    const_init(N, a, 2.5);
    const_init(N, c, 0.0);
    diag_init(N, b, 1.0);

    // declare null pointers for GPU memory
    float* d_a;
    float* d_b;
    float* d_c;

    auto start = std::chrono::high_resolution_clock::now();
    
    // allocate memory on gpu
    CUDA_ERROR_CHECK (cudaMalloc(reinterpret_cast<void**>(&d_a), N*N*sizeof(float)));
    CUDA_ERROR_CHECK (cudaMalloc(reinterpret_cast<void**>(&d_b), N*N*sizeof(float)));
    CUDA_ERROR_CHECK (cudaMalloc(reinterpret_cast<void**>(&d_c), N*N*sizeof(float)));

    // copy data to gpu
    CUDA_ERROR_CHECK (cudaMemcpy(d_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK (cudaMemcpy(d_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice));

    // initialize result variable on gpu
    CUDA_ERROR_CHECK (cudaMemset(d_c, 0.0, N*N*sizeof(float)));

    // call the kernel
    dim3 block(32, 32);
    dim3 grid((N+block.x-1)/block.x, (N+block.y-1)/block.y);

    auto calc_start = std::chrono::high_resolution_clock::now();
    mxm_naive_kernel<<<grid, block>>>(N, d_a, d_b, d_c);

    CUDA_LASTERR();

    // copy data from device to memory
    CUDA_ERROR_CHECK (cudaMemcpy(c, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto elapsed_calc_us = std::chrono::duration_cast<std::chrono::microseconds>(end - calc_start).count();

    // printing the result
    printf("matrix size: %d, elapsed time on gpu: %f ms (with memory copy: %f) \n", N, float(elapsed_calc_us)/1e3, float(elapsed_us)/1e3);

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
    CUDA_ERROR_CHECK (cudaFree(d_a));
    CUDA_ERROR_CHECK (cudaFree(d_b));
    CUDA_ERROR_CHECK (cudaFree(d_c));
    delete[] a, b, c;
}

int main()
{
    for(int i=2; i<2048; i=i*2)
    {
        mxm_test_gpu(i);
    }

    return 0;
}