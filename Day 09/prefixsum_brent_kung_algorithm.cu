#define LOAD_SIZE 32
#include <iostream>
#include <cuda_runtime.h>
// going to code Brent-Kung algorithm
__global__ void prefixsum_kernel(float *A, float *C, int N)
{
    int threadId = threadIdx.x;
    int i = 2 * blockDim.x * blockIdx.x + threadId;

    // Load input data into shared memory for faster access
    // Each thread loads two elements to maximize memory bandwidth utilization
    __shared__ float S_A[LOAD_SIZE]; // shared memory size is 32 floats, so we can load 64 floats in total shared between all threads
    if (i < N)
    {
        S_A[threadId] = A[i];
    }
    if (i + blockDim.x < N)
    {
        S_A[threadId + blockDim.x] = A[i + blockDim.x];
    }
    __syncthreads();

    // Phase 1: Up-sweep (reduction phase) - Build binary tree of partial sums
    // Each iteration doubles the jump distance and reduces active threads by half
    for (int jump = 1; jump <= blockDim.x; jump *= 2)
    {
        __syncthreads(); // Ensure all threads complete previous iteration before proceeding
        int j = jump * 2 * (threadId + 1) - 1; // Calculate index for current thread's operation
        if (j < LOAD_SIZE)
        {
            // Add left child to right child, building partial sums up the tree
            S_A[j] += S_A[j - jump];
        }
    }
    __syncthreads();

    // Phase 2: Down-sweep (distribution phase) - Distribute partial sums down the tree
    // This phase propagates the prefix sums from root to leaves
    for (int jump = LOAD_SIZE / 4; jump >= 1; jump /= 2)
    {
        __syncthreads(); // Synchronize before each sweep iteration
        int j = jump * 2 * (threadId + 1) - 1; // Calculate index for current thread's operation
        if (j < LOAD_SIZE - jump)
        {
            // Propagate sum from parent to right child in the binary tree
            S_A[j + jump] += S_A[j];
        }
        __syncthreads();
    }
    
    // Write results back to global memory
    // Each thread writes back the two elements it originally loaded
    if (i < N)
        C[i] = S_A[threadId];
    if (i < N - blockDim.x)
        C[i + blockDim.x] = S_A[threadId + blockDim.x];
    __syncthreads();
}

void checkCudaError(const char *message)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error (%s): %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

int main()
{
    int N = 10;
    float A[N], C[N];
    for (int i = 0; i < N; i++)
    {
        A[i] = i + 1.0f;
    }
    float *d_A;
    float *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input data to device");
    dim3 dimBlock(32);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);
    prefixsum_kernel<<<dimGrid, dimBlock>>>(d_A, d_C, N);
    checkCudaError("Failed to execute the kernel");
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output data to host");

    cudaFree(d_A);
    cudaFree(d_C);

    // printing the results
    printf("A:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", A[i]);
    }
    printf("C:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", C[i]);
    }
}