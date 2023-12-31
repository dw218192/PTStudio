#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

#include "utils.h"
#include "helpers.cuh"

GLOBAL void incrementKernel(int* d_array, int arraySize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < arraySize) {
		increment(d_array, idx);
	}
}
int main() {
	thrust::device_vector<int> v;
	thrust::host_vector<int> vh{ 12,3,4 };
	v.assign(vh.begin(), vh.end());
	thrust::sort(thrust::device, v.begin(), v.end());

	dim3 grid_dim (div_up(v.size(), static_cast<size_t>(8)), div_up(v.size(), static_cast<size_t>(8)));
	dim3 block_dim (8, 8);

	incrementKernel KERN_PARAM(grid_dim, block_dim)
		(thrust::raw_pointer_cast(v.data()), v.size());

	thrust::copy(v.begin(), v.end(), vh.begin());
	for(size_t i=0; i<vh.size(); ++i) {
		std::cout << vh[i] << " \n"[i == vh.size() - 1];
	}

	return 0;
}