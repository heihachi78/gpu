#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define DATA_LENGTH 16

/*
  __global__
    executed on the device
    callable from the host
    callable from the device with compute compatibility 3
    must return void
*/
__global__ void my_kernel(int* device_data_a, int* device_data_b)
{
  /*
    gridDim   - Dimensions of the grid (how many blocks in each dimension)
    blockDim  - Dimensions of each block (how many threads per block in each dimension)
    blockIdx  - Index of the current block within the grid
    threadIdx - Index of the current thread within its block

    Each is a dim3 struct with .x, .y, .z components. For example, if you launch a kernel
    with 10 blocks of 256 threads each:
      gridDim.x = 10 and blockDim.x = 256
  */
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  device_data_a[tid] += device_data_b[tid];
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

  /*
    cudaMalloc((void**)&device_data_a, DATA_LENGTH * sizeof(int)); allocates memory on the GPU.
    (void**) is a cast to "pointer to void pointer".
    - device_data_a is an int* (pointer to int)
    - &device_data_a is an int** (pointer to pointer to int)
    - cudaMalloc expects a void** parameter
    - (void**) casts &device_data_a from int** to void**

  The function modifies device_data_a to point to the newly allocated GPU memory.
  */
  cudaMalloc((void**)&device_data_a, DATA_LENGTH * sizeof(int));
  cudaMalloc((void**)&device_data_b, DATA_LENGTH * sizeof(int));

  /*
    cudaMemcpyHostToDevice copies data from CPU memory (host) to GPU memory (device).
    It's one of the cudaMemcpyKind enum values used with cudaMemcpy():

    Other common values:
    - cudaMemcpyDeviceToHost - GPU to CPU
    - cudaMemcpyDeviceToDevice - GPU to GPU
    - cudaMemcpyHostToHost - CPU to CPU
  */
  cudaMemcpy(device_data_a, host_data_a, DATA_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_data_b, host_data_b, DATA_LENGTH * sizeof(int), cudaMemcpyHostToDevice);

  /*
    my_kernel<<<grid size, block size, size of required shared memory, index of stream>>>
  */
  my_kernel<<<2, 8>>>(device_data_a, device_data_b);

  cudaMemcpy(host_data_a, device_data_a, DATA_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
  
  for(int i=0;i<DATA_LENGTH;i++)
  {
    printf("%d - %d\n", i, host_data_a[i]);
  }

  cudaFree(device_data_a)
  cudaFree(device_data_b)

  return 0;
}
