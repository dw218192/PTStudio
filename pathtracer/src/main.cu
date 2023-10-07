#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {
	thrust::device_vector<int> v;
	thrust::host_vector<int> vh{ 12,3,4 };
	v.assign(vh.begin(), vh.end());
	thrust::sort(thrust::device, v.begin(), v.end());

	for (size_t i = 0; i < v.size(); ++i) {
		std::cout << v[i] << " \n"[i == v.size() - 1];
	}
	return 0;
}