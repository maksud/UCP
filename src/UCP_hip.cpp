#include <iostream>
#include <hip/hip_runtime.h>

__global__ void sumKernel(const int* a, const int* b, int* c, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) c[i] = a[i] + b[i];
}

int main() {
    const int N = 10;
    int h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    int *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N * sizeof(int));
    hipMalloc(&d_b, N * sizeof(int));
    hipMalloc(&d_c, N * sizeof(int));

    hipMemcpy(d_a, h_a, N * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N * sizeof(int), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(sumKernel, dim3(1), dim3(N), 0, 0, d_a, d_b, d_c, N);

    hipMemcpy(h_c, d_c, N * sizeof(int), hipMemcpyDeviceToHost);

    std::cout << "HIP result: ";
    for (int i = 0; i < N; ++i) std::cout << h_c[i] << " ";
    std::cout << std::endl;

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    return 0;
}

