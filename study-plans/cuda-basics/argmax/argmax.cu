#include <cuda_runtime.h>
#include <float.h>
#include <climits>

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    __shared__ int idx[256];
    __shared__ float max_val[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    max_val[t] = (i < N) ? input[i] : -INFINITY;
    idx[t]     = (i < N) ? t    : INT_MAX;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (t < stride) {
        float other = max_val[t + stride];
        int other_idx = idx[t + stride];

        if (other > max_val[t] ||
            (other == max_val[t] && other_idx < idx[t])) {
            max_val[t] = other;
            idx[t] = other_idx;
        }
    }
    __syncthreads();
}
    if (t == 0) {
        block_vals[blockIdx.x] = max_val[0];
        block_idxs[blockIdx.x] = idx[0];
    }
}

__global__ void argmax_finalize_kernel(
    const float* block_vals,
    const int* block_idxs,
    int* result,
    int num_blocks
) {
    __shared__ float vals[256];
    __shared__ int idx[256];

    int t = threadIdx.x;
    // vals[t] = (t < num_blocks) ? block_vals[t] : -INFINITY;
    // idx[t] = (t < num_blocks)
    //        ? t * 256 + block_idxs[t]  // global index
    //        : INT_MAX;
    // __syncthreads();
    float best = -INFINITY;
    int best_idx = INT_MAX;

    for (int b = t; b < num_blocks; b += blockDim.x) {
        float v = block_vals[b];
        int global_idx = b * 256 + block_idxs[b]; // first-stage index is block-local

        if (v > best || (v == best && global_idx < best_idx)) {
            best = v;
            best_idx = global_idx;
        }
    }

    vals[t] = best;
    idx[t] = best_idx;
    __syncthreads();

    for (int stride = 128; stride > 0; stride /= 2) {
        if (t < stride) {
            int j = t + stride;
            if (vals[j] > vals[t] ||
                (vals[j] == vals[t] && idx[j] < idx[t])) {
                vals[t] = vals[j];
                idx[t] = idx[j];
            }
        }
        __syncthreads();
    }

    if (t == 0) result[0] = idx[0];
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
