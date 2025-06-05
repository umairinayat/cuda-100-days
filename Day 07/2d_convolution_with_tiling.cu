#include <stdio.h>
#include <iostream>

// I'm assuming that the mask and the matrix to be square for simplicity
#define Mask_width 5 // this is the definition of the mask width
#define shared_size (32 + Mask_width - 1) // this is the definition of the shared size
__constant__ float M[Mask_width][Mask_width]; // this is the definition of the mask

__global__ void twod_convolution_kernel(const float *A, float *C, int n) // this is the definition of the kernel
{
    int threadx = threadIdx.x; // this is the definition of the threadx
    int thready = threadIdx.y; // this is the definition of the thready
    int i = blockDim.x * blockIdx.x + threadx; // this is the definition of the i
    int j = blockDim.y * blockIdx.y + thready; // this is the definition of the j

    __shared__ float S_A[shared_size][shared_size]; // this is the definition of the shared memory

    // Load main data
    if ((i < n) && (j < n)) // this is the loading of the main data
    {
        S_A[threadx + Mask_width / 2][thready + Mask_width / 2] = A[i * n + j]; // this is the loading of the main data
    }

    // Load left halo
    if (threadx < Mask_width / 2) // this is the loading of the left halo
    {
        int left_idx = blockIdx.x * blockDim.x - (Mask_width / 2) + threadx; // this is the loading of the left halo
        if (left_idx >= 0 && j < n) // this is the loading of the left halo
        {
            S_A[threadx][thready + Mask_width / 2] = A[left_idx * n + j]; // this is the loading of the left halo
        }
        else
        {
            S_A[threadx][thready + Mask_width / 2] = 0.0f; // this is the loading of the left halo
        }
    }

    // Load right halo
    if (threadx < Mask_width / 2) // this is the loading of the right halo
    {
        int right_idx = blockIdx.x * blockDim.x + blockDim.x + threadx; // this is the loading of the right halo
        if (right_idx < n && j < n) // this is the loading of the right halo
        {
            S_A[threadx + blockDim.x + Mask_width / 2][thready + Mask_width / 2] = A[right_idx * n + j]; // this is the loading of the right halo
        }
        else
        {
            S_A[threadx + blockDim.x + Mask_width / 2][thready + Mask_width / 2] = 0.0f; // this is the loading of the right halo
        }
    }

    // Load top halo
    if (thready < Mask_width / 2) // this is the loading of the top halo
    {
        int top_idy = j - (Mask_width / 2) + thready; // this is the loading of the top halo
        if (top_idy >= 0 && i < n) // this is the loading of the top halo
        {
            S_A[threadx + Mask_width / 2][thready] = A[i * n + top_idy]; // this is the loading of the top halo
        }
        else
        {
            S_A[threadx + Mask_width / 2][thready] = 0.0f; // this is the loading of the top halo
        }
    }

    // Load bottom halo
    if (thready < Mask_width / 2) // this is the loading of the bottom halo
    {
        int bottom_idy = j + blockDim.y + thready; // this is the loading of the bottom halo
        if (bottom_idy < n && i < n) // this is the loading of the bottom halo
        {
            S_A[threadx + Mask_width / 2][thready + blockDim.y + Mask_width / 2] = A[i * n + bottom_idy]; // this is the loading of the bottom halo
        }
        else
        {
            S_A[threadx + Mask_width / 2][thready + blockDim.y + Mask_width / 2] = 0.0f; // this is the loading of the bottom halo
        }
    }

    __syncthreads(); // this is the synchronization of the threads

    if ((i < n) && (j < n)) // this is the convolution operation
    {
        float result = 0.0f; // this is the definition of the result
        for (int k = 0; k < Mask_width; k++) // this is the convolution operation
        {
            for (int x = 0; x < Mask_width; x++) // this is the convolution operation
            {
                result += S_A[threadx + k][thready + x] * M[k][x]; // this is the convolution operation
            }
        }
        C[i * n + j] = result; // this is the storing of the result 
    }
}

void checkCudaError(const char *message)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "%s - CUDA Error: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main() // this is the main function
{
    int n = 10; // this is the definition of the n
    float *h_A = (float *)malloc(n * n * sizeof(float)); // this is the definition of the h_A
    float *h_C = (float *)malloc(n * n * sizeof(float)); // this is the definition of the h_C
    float d_M[Mask_width][Mask_width]; // this is the definition of the d_M

    for (int i = 0; i < Mask_width; i++) // this is the initialization of the mask
    {
        for (int j = 0; j < Mask_width; j++) // this is the initialization of the mask
        {
            d_M[i][j] = 5; // this is the initialization of the mask
        }
    }

    for (int i = 0; i < n; i++) // this is the initialization of the array
    {
        for (int j = 0; j < n; j++) // this is the initialization of the array
        {
            h_A[i * n + j] = 3; // this is the initialization of the array
        }
    }

    float *d_a, *d_c; // this is the definition of the d_a and d_c
    cudaMalloc(&d_a, n * n * sizeof(float)); // this is the allocation of the memory on the device
    cudaMalloc(&d_c, n * n * sizeof(float)); // this is the allocation of the memory on the device
    cudaMemcpy(d_a, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice); // this is the copying of the data from the host to the device
    checkCudaError("Failed to copy input data to device"); // this is the checking of the error
    cudaMemcpyToSymbol(M, d_M, Mask_width * Mask_width * sizeof(float)); // this is the copying of the mask to the device
    checkCudaError("Failed to copy mask data to device"); // this is the checking of the error

    dim3 dimBlock(32, 32); // this is the definition of the block size
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y); // this is the definition of the grid size
    twod_convolution_kernel<<<dimGrid, dimBlock>>>(d_a, d_c, n); // this is the calling of the kernel
    checkCudaError("Failed to execute the kernel"); // this is the checking of the error

    cudaDeviceSynchronize(); // this is the synchronization of the device
    cudaMemcpy(h_C, d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost); // this is the copying of the data from the device to the host
    checkCudaError("Failed to copy output data to host"); // this is the checking of the error

    // Print results
    printf("Results:\n"); // this is the printing of the results
    for (int i = 0; i < n; i++) // this is the printing of the results
    {
        for (int j = 0; j < n; j++) // this is the printing of the results
        {
            printf("%.2f ", h_C[i * n + j]); // this is the printing of the results
        }
        printf("\n"); // this is the printing of the results
    }

    // Clean up
    cudaFree(d_a); // this is the freeing of the memory on the device
    cudaFree(d_c); // this is the freeing of the memory on the device
    free(h_A); // this is the freeing of the memory on the host
    free(h_C); // this is the freeing of the memory on the host

    return 0; // this is the return of the main function    
}
