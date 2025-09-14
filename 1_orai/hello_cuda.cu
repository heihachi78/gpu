#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define DATA_LENGTH 16

__global__ void my_kernel(int* device_data_a, int* device_data_b)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  device_data_a[tid] += device_data_a[tid];
}

int main()
{
  int host_data_a[DATA_LENGTH] = {0};
  int host_data_b[DATA_LENGTH] = {0};
  for(int i=0;i<DATA_LENGTH;i++)
  {
    host_data_a[i] = 1;
    host_data_b[i] = i;
  }
  int* device_data_a;
  int* device_data_b;
  cudaMalloc((void**)&device_data_a, DATA_LENGTH * sizeof(int));
  cudaMalloc((void**)&device_data_b, DATA_LENGTH * sizeof(int));
  cudaMemcpy(device_data_a, host_data_a, DATA_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_data_b, host_data_b, DATA_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
  my_kernel<<<1, DATA_LENGTH>>>(device_data_a, device_data_b);
  cudaMemcpy(host_data_a, device_data_a,  DATA_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
  for(int i=0;i<DATA_LENGTH;i++)
  {
    printf("%d - %d\n", i, host_data_a[i]);
  }
  return 0;
}
