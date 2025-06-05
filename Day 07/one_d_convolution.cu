#include <iostream>
#include <cuda_runtime.h>

#define Mask_width 5

__constant__ float N[Mask_width];

__global__ void oned_convoluation_kernal(const float *A, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        float result = 0.0f;
        for (int k = -1 * Mask_width / 2; k < Mask_width / 2 + 1; k++) // this for loop is used to iterate over the mask width we multiple -1 * mask_width / 2 due to the mask width is 5 and we want to start from the center of the mask
        {
            printf("%.i", k);
            if (i + k >= 0 && i + k < n) // this if statement is used to check if the index is within the bounds of the array
            {

                result += A[i + k] * N[k + Mask_width / 2]; // this is the convolution operation we multiply the value of the mask with the value of the array and add it to the result
            }
        }
        C[i] = result;
    }
}

void check_cuda_error(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

int main()
{

    int n = 10;
    float A[n], C[n];
    float d_M[Mask_width];

    for (int i = 0; i < Mask_width; i++) // this for loop is used to initialize the mask
    {
        d_M[i] = i;
    }
    for (int i = 0; i < n; i++) // this for loop is used to initialize the array
    {
        A[i] = i;
    }

    float *d_a, *d_c;                                              // this is the pointer to the device memory
    cudaMalloc(&d_a, n * sizeof(float));                           // this is the allocation of the memory on the device
    cudaMalloc(&d_c, n * sizeof(float));                           // this is the allocation of the memory on the device
    cudaMemcpy(d_a, A, n * sizeof(float), cudaMemcpyHostToDevice); // this is the copying of the data from the host to the device
    checkCudaError("Failed to copy input data to device");
    cudaMemcpyToSymbol(M, d_M, Mask_width * sizeof(float));        // this is the copying of the mask to the device
    checkCudaError("Failed to copy mask data to device");          // this is the checking of the error
    dim3 dimBlock(32);                                             // this is the definition of the block size
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);               // this is the definition of the grid size
    oned_convolution_kernel<<<dimGrid, dimBlock>>>(d_a, d_c, n);   // this is the calling of the kernel
    checkCudaError("Failed to execute the kernel");                // this is the checking of the error
    cudaDeviceSynchronize();                                       // this is the synchronization of the device
    cudaMemcpy(C, d_c, n * sizeof(float), cudaMemcpyDeviceToHost); // this is the copying of the data from the device to the host
    checkCudaError("Failed to copy output data to host");          // this is the checking of the error
    cudaFree(d_a);                                                 // this is the freeing of the memory on the device
    cudaFree(d_c);                                                 // this is the freeing of the memory on the device

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

        printf("%.2f ", d_M[i]);
    }
    printf("\n");
    printf("\nC:\n");
    for (int i = 0; i < n; i++)
    {

        printf("%.2f ", C[i]);
    }
    printf("\n");
}