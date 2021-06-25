#include "cuda_kernels.h"


/* CUDA device code for vector operation kernels */

//------------------------------------------------------------------------
/* Addition kernel:
	Input:
		- a, b, c are double pointers of address
		- N is vector length
*/

__global__
void VectorKernels::add(const int* a, const int* b, int* c, const int N)
{
	// Calculate global thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Perform addition of one element if thread ID less than vector length
	if (tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
}


//------------------------------------------------------------------------
/* Multiplication kernel:
	Input:
		- a, b, c are double pointers of address
		- N is vector length
*/

__global__
void VectorKernels::multiply(const int* a, const int* b, int* c, const int N)
{
	// Calculate global thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Perform addition of one element if thread ID less than vector length
	if (tid < N)
	{
		c[tid] = a[tid] * b[tid];
	}
}