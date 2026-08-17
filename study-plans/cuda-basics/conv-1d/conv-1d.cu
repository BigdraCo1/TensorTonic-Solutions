#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int kN) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N - kN + 1) {
        float sum = 0;
        for (int j = 0; j < kN; j++) {
            sum += input[index + j] * kernel[j];
        }
        output[index] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threads = 256;
    dim3 blocks((outN + 255) / 256);
    conv1d_kernel<<<blocks, threads>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}
