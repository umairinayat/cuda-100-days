#include <iostream>

__global__ void partialSum(const float *input, float *output, int N)
{
    // Shared memory
    extern __shared__ int sharedMemory[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 2 + tid;

    if (index < n)
    {
        // Load input into shared memory and optimize the loading to do coalescing
        sharedMemory[tid] = input[index] + input[index + blockDim.x];
        __syncthreads(); // Ensure all threads have loaded their data into shared memory

        // Perform inclusive scan in shared memory
        for (int stride = 1; stride < blockDim.x; stride *= 2) // multiply by 2 each iteration because we are doubling the stride
        // the stride is the distance between elements to be summed and we are doubling it each iteration because we are summing pairs of elements
        // This loop performs the inclusive scan by summing elements in shared memory
        {
            int temp = 0;
            if (tid >= stride) // Check if the thread index is greater than or equal to stride
            {
                temp = sharedMemory[tid - stride]; // Get the value from shared memory at the position of stride before this thread
            }
            __syncthreads();           // Ensure all threads have completed the previous step before proceeding
            sharedMemory[tid] += temp; // Add the value from the previous stride to the current value in shared memory
            __syncthreads();           // Ensure all threads have completed the addition before proceeding
        }

        // Write result to global memory
        output[index] = sharedMemory[tid]; // Write the result back to global memory global memory is used for storing the final result
        if (index + blockDim.x < N)        // Check if the next element is within bounds
        {
            output[index + blockDim.x] = sharedMemory[tid]; // Write the result for the next element
        }
    }
}

int main()
{
    const int N = 16;
    const int blockSize = 8;

    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int h_output[N];

    int *d_input, *d_output;
    size_t size = N * sizeof(int); // Size in bytes for N integers size_t is used for size in bytes size_t is an unsigned integer type that is used to represent the size of an object in bytes
    // Allocate memory on the device

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size); // Allocate memory for the output on the device
    // Copy input data from host to device

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    partialSumKernel<<<N / blockSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, N); // Launch the kernel with N / blockSize blocks and blockSize threads per block why we are N / blockSize blocks? because we are processing N elements and each block can process blockSize elements, so we need N / blockSize blocks to process all elements
    // Copy output data from device to host

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Input: "); // Print the input array
    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_input[i]);
    } // Print the input array
    printf("\nOutput: ");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_output[i]);
    } // Print the output array
    printf("\n");

    cudaFree(d_input); // Free device memory
    cudaFree(d_output); // Free device memory

    return 0;
}