#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int outputH = H - kH + 1;
    int outputW = W - kW + 1;
    if (i < outputW && j < outputH) {
        float res = 0.0f;
        for (int b = 0; b < kH; b++) {
            for (int a = 0; a < kW; a++) {
                float input_val =
                    input[(j + b) * W + (i + a)];

                float kernel_val =
                    kernel[b * kW + a];

                res += input_val * kernel_val;
            }
        }
        output[j * outputW + i] = res;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    conv2d_kernel<<<blocks, threads>>>(input, kernel, output, H, W, kH, kW);
    cudaDeviceSynchronize();
}
