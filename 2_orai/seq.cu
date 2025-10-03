#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "prelude.h"

void mxm_serial(int N, float* a, float* b, float* c)
{

}

int mxm_test_serial(int N)
{
    // heap allocate memory
    float* a = new float[N*N];
    float* b = new float[N*N];
    float* c = new float[N*N];

    // call our function
    mxm_serial(N, a, b, c);

    // free memory
    delete[] a, b, c;
}

int main()
{
    return 0;
}