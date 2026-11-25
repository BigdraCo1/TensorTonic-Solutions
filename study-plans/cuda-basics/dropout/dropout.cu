#include <cuda_runtime.h>

__global__ void dropout_kernel(const float* input, const float* mask, float* output, float p, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float scale = 1.0f / (1.0f - p);
    if (i < N) output[i] = input[i] * mask[i] * scale;
}

extern "C" void solve(const float* input, const float* mask, float* output, float p, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    dropout_kernel<<<blocks, threads>>>(input, mask, output, p, N);
    cudaDeviceSynchronize();
}