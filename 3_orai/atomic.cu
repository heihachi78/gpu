#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "prelude.h"
#include <time.h>

void random_array(float* arr, int arr_len, float min, float max, unsigned seed)
{
    srand(seed);
    for(int i=0; i<arr_len; i++)
    {
        arr[i] = ((float)rand() / ((float)RAND_MAX + 1.0f)) * (max - min) + min;
    }
}

template <typename T>
void const_array(T* arr, int arr_len, T init_val)
{
    for(int i=0; i<arr_len; i++)
    {
        arr[i] = init_val;
    }
}

__device__ __host__ int calc_bin(float item, float min, float max, float bin_width)
{
    item -= min;
    max -= min;
    return (int)(item / bin_width);
}

void histogram_cpu(float* arr, int arr_len, float min, float max, int* bins, int num_bins)
{
    float bin_width = (max - min) / (float)num_bins;
    for(int i=0; i<arr_len; i++)
    {
        bins[calc_bin(arr[i], min, max, bin_width)]++;
    }
}

void print_bins(int* bins, int num_bins)
{
    printf("HISTOGRAM\n");
    for(int i=0; i<num_bins; i++)
    {
        printf("\t%4d\n", bins[i]);
    }
}

__global__ void histogram_kernel_naive(float* arr, int arr_len, float min, float max, int* bins, int num_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= arr_len)
    {
        return;
    }
    float bin_width = (max - min) / (float)num_bins;
    atomicAdd(bins + calc_bin(arr[idx], min, max, bin_width), 1);
}

int main()
{
    int arr_len = 10;
    int num_bins = 2;
    float min = 0.0f;
    float max = 10.0f;
    float* arr = new float[arr_len];
    int* bins = new int[num_bins];

    random_array(arr, arr_len, min, max, 19780428);
    const_array(bins, num_bins, 0);

    histogram_cpu(arr, arr_len, min, max, bins, num_bins);
    print_bins(bins, num_bins);

    delete[] arr;
    delete[] bins;

    return 0;
}