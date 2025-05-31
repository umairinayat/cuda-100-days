#include <iostream>
#include <cmath>

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
    for(int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    // Allocate GPU memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = ceil(N / (float)blockSize);
    vectAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Copy result back to host
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Vector Addition Results:\n";
    for(int i = 0; i < N; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }
    
    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
