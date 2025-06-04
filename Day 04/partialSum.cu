#include <iostream>
#include <cuda_runtime.h>

__global__ void partialSum(const float *input, float *output, int N)
{
    extern __shared__ float sharedMemory[]; // Changed to float
    
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid; // Simplified indexing
    
    // Load input into shared memory with bounds checking
    if (index < N) {
        sharedMemory[tid] = input[index];
    } else {
        sharedMemory[tid] = 0.0f; // Initialize to zero if out of bounds
    }
    __syncthreads();
    
    // Perform inclusive scan (prefix sum)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (tid >= stride) {
            temp = sharedMemory[tid - stride];
        }
        __syncthreads();
        sharedMemory[tid] += temp;
        __syncthreads();
    }
    
    // Write result back to global memory
    if (index < N) {
        output[index] = sharedMemory[tid];
    }
}

int main()
{
    const int N = 16;
    const int blockSize = 8;
    
    // Changed to float arrays
    float h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_output[N];
    
    float *d_input, *d_output;
    size_t size = N * sizeof(float); // Changed to sizeof(float)
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Launch kernel with proper shared memory size for floats
    partialSum<<<(N + blockSize - 1) / blockSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    printf("Input: ");
    for (int i = 0; i < N; i++) {
        printf("%.0f ", h_input[i]);
    }
    printf("\nOutput: ");
    for (int i = 0; i < N; i++) {
        printf("%.0f ", h_output[i]);
    }
    printf("\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
