#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float a = 0;
        float max_num = -INFINITY;
        for (int j = 0; j < N; j++) {
            if (input[j] > max_num) {
                max_num = input[j];
            }
        }
        for (int k = 0; k < N; k++) {
            a += exp(input[k] - max_num);
        }
        output[i] = exp(input[i] - max_num)/a;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}