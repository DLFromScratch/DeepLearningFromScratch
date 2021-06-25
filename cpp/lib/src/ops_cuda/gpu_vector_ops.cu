#include <stdexcept>

#include "cuda_kernels.h"
#include "vector_ops.h"


/* CUDA host code for GPU vector operations */

//------------------------------------------------------------------------
/* Addition operation:
	Input:
		- a, b input std::vector
		- c is output std::vector
*/

void GPUVectorOps::add(
	const std::vector<float>& a,
	const std::vector<float>& b,
	std::vector<float>& c
)
{
	// TODO: check if this is faster than vector::size()
	const int LENGTH{ static_cast<int>(a.size()) };
	const std::size_t vectorSize{ LENGTH * sizeof(float) };
	const int NUM_THREADS{ 1024 };
	const int NUM_BLOCKS{ LENGTH + NUM_THREADS - 1 };

	if (LENGTH != b.size() || LENGTH != c.size())
	{
		throw std::length_error("Incompatible vector shapes");
	}

	// Declare pointers for CUDA device
	int* d_a = NULL;
	int* d_b = NULL;
	int* d_c = NULL;

	// CUDA Malloc takes address of the pointers to reserve space on GPU
	cudaMalloc(&d_a, vectorSize);
	cudaMalloc(&d_b, vectorSize);
	cudaMalloc(&d_c, vectorSize);

	// Copy vectors to GPU and launch kernel
	cudaMemcpy(d_a, a.data(), vectorSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), vectorSize, cudaMemcpyHostToDevice);
	VectorKernels::add <<<NUM_BLOCKS, NUM_THREADS >>> (d_a, d_b, d_c, LENGTH);

	// Copy results to CPU
	cudaMemcpy(c.data(), d_c, vectorSize, cudaMemcpyDeviceToHost);
	cudaFree(&d_a);
	cudaFree(&d_b);
	cudaFree(&d_c);
}

//------------------------------------------------------------------------
/* Multiplication operation:
	Input:
		- a, b input std::vector
		- c is output std::vector
*/

void GPUVectorOps::multiply(
	const std::vector<float>& a,
	const std::vector<float>& b,
	std::vector<float>& c
)
{
	// TODO: check if this is faster than vector::size()
	const int LENGTH{ static_cast<int>(a.size()) };
	const std::size_t vectorSize{ LENGTH * sizeof(float) };
	const int NUM_THREADS{ 1024 };
	const int NUM_BLOCKS{ LENGTH + NUM_THREADS - 1 };

	if (LENGTH != b.size() || LENGTH != c.size())
	{
		throw std::length_error("Incompatible vector shapes");
	}

	// Declare pointers for CUDA device
	int* d_a = NULL;
	int* d_b = NULL;
	int* d_c = NULL;

	// CUDA Malloc takes address of the pointers to reserve space on GPU
	cudaMalloc(&d_a, vectorSize);
	cudaMalloc(&d_b, vectorSize);
	cudaMalloc(&d_c, vectorSize);

	// Copy vectors to GPU and launch kernel
	cudaMemcpy(d_a, a.data(), vectorSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), vectorSize, cudaMemcpyHostToDevice);
	VectorKernels::add <<<NUM_BLOCKS, NUM_THREADS >>> (d_a, d_b, d_c, LENGTH);

	// Copy results to CPU and free memory
	cudaMemcpy(c.data(), d_c, vectorSize, cudaMemcpyDeviceToHost);
	cudaFree(&d_a);
	cudaFree(&d_b);
	cudaFree(&d_c);
}

//------------------------------------------------------------------------
/* Combined addition and multiplication operation:
	Input:
		- a, b: input std::vector
		- c: output std::vector
*/

void GPUVectorOps::addMultiply(
	const std::vector<float>& a,
	const std::vector<float>& b,
	std::vector<float>& c
)
{
	// TODO: check if this is faster than vector::size()
	const int LENGTH{ static_cast<int>(a.size()) };
	const std::size_t vectorSize{ LENGTH * sizeof(float) };
	const int NUM_THREADS{ 1024 };
	const int NUM_BLOCKS{ LENGTH + NUM_THREADS - 1 };

	if (LENGTH != b.size() || LENGTH != c.size())
	{
		throw std::length_error("Incompatible vector shapes");
	}

	// Declare pointers for CUDA device
	int* d_a = NULL;
	int* d_b = NULL;
	int* d_c = NULL;

	// CUDA Malloc takes address of the pointers to reserve space on GPU
	cudaMalloc(&d_a, vectorSize);
	cudaMalloc(&d_b, vectorSize);
	cudaMalloc(&d_c, vectorSize);

	// Copy vectors to GPU and launch kernel
	cudaMemcpy(d_a, a.data(), vectorSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), vectorSize, cudaMemcpyHostToDevice);
	VectorKernels::add <<<NUM_BLOCKS, NUM_THREADS >>> (d_a, d_b, d_c, LENGTH);
	cudaDeviceSynchronize(); // Is this actually necessary here?
	VectorKernels::multiply <<<NUM_BLOCKS, NUM_THREADS >>> (d_a, d_b, d_c, LENGTH);

	// Copy results to CPU and free memory
	cudaMemcpy(c.data(), d_c, vectorSize, cudaMemcpyDeviceToHost);
	cudaFree(&d_a);
	cudaFree(&d_b);
	cudaFree(&d_c);
}

