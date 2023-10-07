#include "include/helpers.cuh"

HOST DEVICE void increment(int* arr, int idx) {
    ++arr[idx];
}
