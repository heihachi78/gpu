/*
  variable length vector
*/

#include <stdio.h>         // Standard I/O functions (printf)
#include <cuda.h>          // CUDA runtime API
#include <cuda_runtime.h>  // Additional CUDA runtime functions
#include "prelude.h"       // for error handling

#define DATA_LENGTH 11     // Size of arrays to process

/*
  __global__
    executed on the device
    callable from the host
    callable from the device with compute compatibility 3
    must return void
*/
__global__ void my_kernel(int N, int* device_data_a, int* device_data_b)
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
  int tid = blockDim.x * blockIdx.x + threadIdx.x; // Calculate global thread ID
  if (tid >= N) 
  {
    return;
  }
  device_data_a[tid] += device_data_b[tid]; // Add corresponding elements
}

int main()
{
  // Initialize arrays on CPU (host)
  int host_data_a[DATA_LENGTH] = {0}; // Will store result
  int host_data_b[DATA_LENGTH] = {0}; // Input array

  // Fill arrays with initial values
  for(int i=0;i<DATA_LENGTH;i++)
  {
    host_data_a[i] = 1; // Initialize to 1
    host_data_b[i] = i; // Initialize to index value
  }

  // Declare pointers for GPU memory
  int* device_data_a; // Will point to GPU memory for array A
  int* device_data_b; // Will point to GPU memory for array B

  /*
    cudaMalloc((void**)&device_data_a, DATA_LENGTH * sizeof(int)); allocates memory on the GPU.
    (void**) is a cast to "pointer to void pointer".
    - device_data_a is an int* (pointer to int)
    - &device_data_a is an int** (pointer to pointer to int)
    - cudaMalloc expects a void** parameter
    - (void**) casts &device_data_a from int** to void**

  The function modifies device_data_a to point to the newly allocated GPU memory.
  */
  // Allocate memory on GPU
  CUDA_ERROR_CHECK (cudaMalloc((void**)&device_data_a, DATA_LENGTH * sizeof(int))); // Allocate GPU memory for array A
  CUDA_ERROR_CHECK (cudaMalloc((void**)&device_data_b, DATA_LENGTH * sizeof(int))); // Allocate GPU memory for array B

  /*
    cudaMemcpyHostToDevice copies data from CPU memory (host) to GPU memory (device).
    It's one of the cudaMemcpyKind enum values used with cudaMemcpy():

    Other common values:
    - cudaMemcpyDeviceToHost - GPU to CPU
    - cudaMemcpyDeviceToDevice - GPU to GPU
    - cudaMemcpyHostToHost - CPU to CPU
  */
  // Copy data from CPU to GPU
  CUDA_ERROR_CHECK (cudaMemcpy(device_data_a, host_data_a, DATA_LENGTH * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK (cudaMemcpy(device_data_b, host_data_b, DATA_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

  /*
    my_kernel<<<grid size, block size, size of required shared memory, index of stream>>>
    Launch kernel with 2 blocks of 8 threads each (16 threads total)

    While grid size * block size = DATA_LENGTH this example will run correctly.
    So 1, 16 or 2, 8 or 4, 4 or 8, 2 or 16, 1 works.
  */

  // calculate block and grid sizes dynamicly
  dim3 block(4);
  dim3 grid((DATA_LENGTH + block.x - 1) / block.x);

  my_kernel<<<grid, block>>>(DATA_LENGTH, device_data_a, device_data_b);

  // Copy result back from GPU to CPU
  CUDA_ERROR_CHECK (cudaMemcpy(host_data_a, device_data_a, DATA_LENGTH * sizeof(int), cudaMemcpyDeviceToHost));

  // Print results
  for(int i=0; i<DATA_LENGTH; i++)
  {
    printf("%d - %d\n", i, host_data_a[i]); // Print index and result
  }

  // Free GPU memory
  CUDA_ERROR_CHECK (cudaFree(device_data_a));
  CUDA_ERROR_CHECK (cudaFree(device_data_b));

  return 0;
}
