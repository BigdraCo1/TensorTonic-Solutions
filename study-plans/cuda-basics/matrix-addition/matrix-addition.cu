#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < N && j < M) C[i + j*N] = A[i + j*N] + B[i + j*N];
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
