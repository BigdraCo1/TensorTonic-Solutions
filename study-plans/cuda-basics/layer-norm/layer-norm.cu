#include <cuda_runtime.h>
#include <math.h>


__device__ void mean_variance_kernel(const float* input, float* mean_out, float* var_out, int N) {
    __shared__ float shared[256];
    __shared__ float shared_sq[256];
    unsigned int tid = threadIdx.x;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (std::size_t col = tid; col < N; col += blockDim.x) {
        const float x = input[col];
        sum += x;
        sum_sq += x * x;
    }
    shared[tid] = sum;
    shared_sq[tid] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
            shared_sq[tid] += shared_sq[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float inv_n = 1.0f / static_cast<float>(N);
        const float mean = shared[0] * inv_n;
        *mean_out = mean;
        *var_out = shared_sq[0] * inv_n - mean * mean;
    }
    __syncthreads();
}


__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    const unsigned int row = blockIdx.x;
    if (row >= static_cast<unsigned int>(M)) return;

    const std::size_t n = N;
    const std::size_t offset = row * n;
    const float* row_input = input + offset;
    float* row_out = output + offset;

    __shared__ float mean;
    __shared__ float var;
    mean_variance_kernel(row_input, &mean, &var, N);
    const float inv_std = rsqrtf(fmaxf(var, 0.0f) + eps);
    for (std::size_t col = threadIdx.x; col < N; col += blockDim.x){
    const float normalized = (row_input[col] - mean) * inv_std;
    row_out[col] = gamma[col] * normalized + beta[col];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
