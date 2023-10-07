#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

#include "../include/utils.h"
#include "../include/helpers.cuh"

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

	dim3 grid_dim (DIV_UP(v.size(), 8), DIV_UP(v.size(), 8));
	dim3 block_dim (8, 8);
	incrementKernel<<<grid_dim, block_dim >>>(thrust::raw_pointer_cast(v.data()), v.size());

	thrust::copy(v.begin(), v.end(), vh.begin());
	for(size_t i=0; i<vh.size(); ++i) {
		std::cout << vh[i] << " \n"[i == vh.size() - 1];
	}

	return 0;
}