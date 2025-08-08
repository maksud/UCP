#include <iostream>
#include <cuda_runtime.h>

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
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    sumKernel<<<1, N>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "CUDA result: ";
    for (int i = 0; i < N; ++i) std::cout << h_c[i] << " ";
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}

