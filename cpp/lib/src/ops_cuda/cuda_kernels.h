#pragma once


/* CUDA Kernels for vector operations */

namespace VectorKernels
{
	__global__ void add(const int*, const int*, int*, const int);
	__global__ void multiply(const int*, const int*, int*, const int);
}
