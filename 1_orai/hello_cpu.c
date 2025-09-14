#include <stdio.h>         // Standard I/O functions (printf)

#define DATA_LENGTH 16     // Size of arrays to process

/*
  CPU version of the kernel - just a regular function
  No __global__ needed since it runs on CPU
  Takes same parameters as GPU version
*/
void my_kernel_cpu(int* data_a, int* data_b)
{
  // Loop through all elements (no parallel execution)
  for(int tid = 0; tid < DATA_LENGTH; tid++)
  {
    data_a[tid] += data_b[tid]; // Add corresponding elements
  }
}

int main()
{
  // Initialize arrays on CPU (host) - same as GPU version
  int host_data_a[DATA_LENGTH] = {0}; // Will store result
  int host_data_b[DATA_LENGTH] = {0}; // Input array

  // Fill arrays with initial values - same as GPU version
  for(int i=0;i<DATA_LENGTH;i++)
  {
    host_data_a[i] = 1; // Initialize to 1
    host_data_b[i] = i; // Initialize to index value
  }

  // No GPU memory allocation needed - data stays on CPU
  // No cudaMalloc calls

  // No memory copying needed - data already on CPU
  // No cudaMemcpy calls

  /*
    Call CPU function directly - no kernel launch syntax <<<>>>
    Pass arrays directly (no device pointers needed)
  */
  my_kernel_cpu(host_data_a, host_data_b);

  // No memory copying back needed - result already in host_data_a

  // Print results - same as GPU version
  for(int i=0;i<DATA_LENGTH;i++)
  {
    printf("%d - %d\n", i, host_data_a[i]); // Print index and result
  }

  // No GPU memory to free - no cudaFree calls

  return 0;
}