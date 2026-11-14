#include <cuda_runtime.h>

__global__ void mean_variance_kernel(const float* input, float* mean_out, float* var_out, int N) {
    __shared__ float shared[256];
    __shared__ float shared_sq[256];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    shared[tid] = (idx < N) ? input[idx] : 0.0f;
    shared_sq[tid] = (idx < N) ? input[idx] * input[idx] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
            shared_sq[tid] += shared_sq[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float inv_n = 1.0f / N;
        atomicAdd(mean_out, shared[0] * inv_n);
        atomicAdd(var_out, shared_sq[0] * inv_n - mean_out[0] * mean_out[0]);
    }
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(mean_out, 0, sizeof(float));
    cudaMemset(var_out, 0, sizeof(float));
    mean_variance_kernel<<<blocks, threads>>>(input, mean_out, var_out, N);
    cudaDeviceSynchronize();
}
