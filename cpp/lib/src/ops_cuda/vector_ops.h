#pragma once

#include <vector>


/* Public functions for vector addition, multiplication etc. using CPU or GPU */

namespace GPUVectorOps
{
	void add(
		const std::vector<float>&,
		const std::vector<float>&,
		std::vector<float>&
	);

	void multiply(
		const std::vector<float>&,
		const std::vector<float>&,
		std::vector<float>&
	);

	void addMultiply(
		const std::vector<float>&,
		const std::vector<float>&,
		std::vector<float>&
	);
}


namespace CPUVectorOps
{
	void add(
		const std::vector<float>&,
		const std::vector<float>&,
		std::vector<float>&
	);

	void multiply(
		const std::vector<float>&,
		const std::vector<float>&,
		std::vector<float>&
	);
}

