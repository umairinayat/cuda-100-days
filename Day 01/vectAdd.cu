#include <iostream>

__global__ void vectAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int N = 10;
    float a[N], b[N], c[N];
    float *d_a, *d_b, *d_c;

    // Initialize vectors a and b
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = ceil(N / (float)blockSize);
    vectAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}