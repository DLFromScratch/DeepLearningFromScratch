#include "vector_ops.h"
#include <iostream>

/* Code for CPU vector operations */

//------------------------------------------------------------------------
/* Addition operation:
	Input:
		- a, b input std::vector
		- c is output std::vector
*/

void CPUVectorOps::add(
	const std::vector<float>& a,
	const std::vector<float>& b,
	std::vector<float>& c
)
{
	if (a.size() != b.size() || a.size() != c.size())
	{
		throw std::length_error("Incompatible vector shapes");
	}

	for (int i{ 0 }; i < a.size(); ++i)
	{
		c[i] = a[i] + b[i];
	}
}

//------------------------------------------------------------------------
/* Multiplication operation:
	Input:
		- a, b input std::vector
		- c is output std::vector
*/

void CPUVectorOps::multiply(
	const std::vector<float>& a,
	const std::vector<float>& b,
	std::vector<float>& c
)
{
	if (a.size() != b.size() || a.size() != c.size())
	{
		throw std::length_error("Incompatible vector shapes");
	}

	for (int i{ 0 }; i < a.size(); ++i)
	{
		c[i] = a[i] + b[i];
	}
}