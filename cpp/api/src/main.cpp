#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "vector_ops.h"


/* Test routine for CUDA ops */

int main()
{
	constexpr int LENGTH{ 50000000 };
	constexpr int LOWER{ 0 };
	constexpr int UPPER{ 100 };

	// Random uniform
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<float> dist(LOWER, UPPER);

	std::vector<float> vec1;
	std::vector<float> vec2;
	std::vector<float> wrongVec;
	std::vector<float> result(LENGTH);

	vec1.reserve(LENGTH);
	vec2.reserve(LENGTH);
	wrongVec.reserve(LENGTH - 1);

	for (int i{ 0 }; i < LENGTH; ++i)
	{
		vec1.push_back(dist(e2));
		vec2.push_back(dist(e2));
	}

	for (int i{ 0 }; i < LENGTH - 1; ++i)
	{
		wrongVec.push_back(dist(e2));
	}

	try
	{
		GPUVectorOps::add(vec1, wrongVec, result);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	// Elapsed time for CPU ops
	auto startTime = std::chrono::steady_clock::now();

	try
	{
		CPUVectorOps::add(vec1, vec2, result);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	try
	{
		CPUVectorOps::multiply(vec1, vec2, result);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	auto endTime = std::chrono::steady_clock::now();
	double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
	std::cout << "CPU separate addition and multiplication: " << elapsedTime << " us" << std::endl;

	// Elapsed time for separate GPU ops
	startTime = std::chrono::steady_clock::now();

	try
	{
		GPUVectorOps::add(vec1, vec2, result);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	try
	{
		GPUVectorOps::multiply(vec1, vec2, result);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	endTime = std::chrono::steady_clock::now();
	elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
	std::cout << "GPU separate addition and multiplication: " << elapsedTime << " us" << std::endl;

	// Elapsed time for combined GPU ops
	startTime = std::chrono::steady_clock::now();

	try
	{
		GPUVectorOps::addMultiply(vec1, vec2, result);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	endTime = std::chrono::steady_clock::now();
	elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
	std::cout << "GPU combined addition and multiplication: " << elapsedTime << " us" << std::endl;

	system("pause");
	return 0;
}