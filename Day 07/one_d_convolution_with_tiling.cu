#include <iostream>
#include <cuda_runtime.h>
#define Mask_width 5
__constant__ float M[Mask_width];

__global__ void oned_convolution_tiling_kernel(const float *A, float *C, int n)
{
    int threadId = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadId;

    __shared__ float S_A[32 + Mask_width - 1]; // this is the definition of the shared memory we add 32 to the mask width to avoid the out of bounds error and we are doing this because we are using the block size of 32

    // Load main data
    if (i < n)
    {
        S_A[threadId + Mask_width / 2] = A[i]; // this is the loading of the main data  
    }

    // Load left halo
    if (threadId < Mask_width / 2) // this is the loading of the left halo
    {
        int left_idx = blockIdx.x * blockDim.x - (Mask_width / 2) + threadId; // this is the loading of the left halo
        if (left_idx >= 0)
        {
            S_A[threadId] = A[left_idx]; // this is the loading of the left halo
        }
        else
        {
            S_A[threadId] = 0.0f; // this is the loading of the left halo
        }
    }

    // Load right halo
    if (threadId < Mask_width / 2) // this is the loading of the right halo
    {
        int right_idx = blockIdx.x * blockDim.x + blockDim.x + threadId; // this is the loading of the right halo
        if (right_idx < n)
        {
            S_A[threadId + blockDim.x + Mask_width / 2] = A[right_idx]; // this is the loading of the right halo
        }
        else
        {
            S_A[threadId + blockDim.x + Mask_width / 2] = 0.0f; // this is the loading of the right halo
        }
    }

    __syncthreads(); // this is the synchronization of the threads for the shared memory syncthreads is used when we are using the shared memory and this is used to make sure that the threads are synchronized

    if (i < n)
    {
        float result = 0.0f;
        for (int k = 0; k < Mask_width; k++) // this is the convolution operation
        {
            int idx = threadId + k;
            if ((i + k - Mask_width / 2) >= 0 && (i + k - Mask_width / 2) < n) // this is the convolution operation
            {
                result += S_A[idx] * M[k]; // this is the convolution operation
            }
        }
        C[i] = result; // this is the storing of the result 
    }
}

// Host function to check for CUDA errors
void checkCudaError(const char *message)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{

    int n = 10;
    float A[n], C[n];
    float d_M[Mask_width];

    for (int i = 0; i < Mask_width; i++) // this is the initialization of the mask
    {
        d_M[i] = i;
    }
    for (int i = 0; i < n; i++) // this is the initialization of the array
    {
        A[i] = i;
    }

    float *d_a, *d_c;
    cudaMalloc(&d_a, n * sizeof(float)); // this is the allocation of the memory on the device     

    cudaMalloc(&d_c, n * sizeof(float)); // this is the allocation of the memory on the device
    
    cudaMemcpy(d_a, A, n * sizeof(float), cudaMemcpyHostToDevice); // this is the copying of the data from the host to the device
    
    checkCudaError("Failed to copy input data to device"); // this is the checking of the error
    
    cudaMemcpyToSymbol(M, d_M, Mask_width * sizeof(float)); // this is the copying of the mask to the device
    
    checkCudaError("Failed to copy mask data to device"); // this is the checking of the error
    
    dim3 dimBlock(32); // this is the definition of the block size
    
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x); // this is the definition of the grid size
    
    oned_convolution_tiling_kernel<<<dimGrid, dimBlock>>>(d_a, d_c, n); // this is the calling of the kernel
    
    checkCudaError("Failed to execute the kernel"); // this is the checking of the error
    
    cudaDeviceSynchronize(); // this is the synchronization of the device
    
    cudaMemcpy(C, d_c, n * sizeof(float), cudaMemcpyDeviceToHost); // this is the copying of the data from the device to the host
    
    
    checkCudaError("Failed to copy output data to host"); // this is the checking of the error
    
    cudaFree(d_a); // this is the freeing of the memory on the device
    
    cudaFree(d_c); // this is the freeing of the memory on the device

    // printing the results
    printf("A:\n"); // this is the printing of the array
    for (int i = 0; i < n; i++)
    {
        printf("%.2f ", A[i]); // this is the printing of the array
    }
    printf("\n");
    printf("\nd_m:\n"); // this is the printing of the mask
    for (int i = 0; i < Mask_width; i++)
    {

        printf("%.2f ", d_M[i]); // this is the printing of the mask
    }
    printf("\n");
    printf("\nC:\n"); // this is the printing of the result
    for (int i = 0; i < n; i++)
    {

        printf("%.2f ", C[i]); // this is the printing of the result    
    }
    printf("\n");

    return 0;
}