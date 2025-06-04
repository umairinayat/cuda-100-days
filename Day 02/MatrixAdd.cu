#include <iostream>

__global__ void matrixAdd_C(const float *A, const float *B,int N, float *C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = A[i * N + j] + B[i * N + j];
        }
        return;
    }
}

__global__ void matrixAdd_B ( const float *A, const float *C, float *B, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i < N){
        for(int j = 0 ; j < N ; j++)
        {
            B[i * N + j] = C[i * N + j] - A[i * N + j];

        }
        return;
    }
}

__global__ void MatrixAdd_D(const float* A, const float* B, float* C, int N) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < N)  {
        for(int i=0;i<N;i++){

          C[i*N+j] = A[i*N+j] + B[i*N+j];

        }

    }

}


int main()
{
    const int N = 10;
    float *A, *B, *C;


    // initialize the input matrices
    A = (float *)malloc( N*N* sizeof(float));
    B = (float *)malloc(N*N* sizeof(float));
    C = (float *)malloc(N*N * sizeof(float));


    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f; // Initialize B with some values
            C[i * N + j] = 0.0f;
        }
    }

    float *d_a, *d_b,*d_c;
    cudaMalloc((void **)&d_a,N*N*sizeof(float)); // Allocate memory for A on the GPU
    cudaMalloc((void **)&d_b,N*N*sizeof(float)); // Allocate memory for B on the GPU
    cudaMalloc((void **)&d_c,N*N*sizeof(float)); // Allocate memory for C on the GPU
    cudaMemcpy(d_a,A,N*N*sizeof(float),cudaMemcpyHostToDevice); // Copy A to the GPU
    cudaMemcpy(d_b,B,N*N*sizeof(float),cudaMemcpyHostToDevice); // Copy B to the GPU

    dim3 dimBlock(32, 16); // Define block size
    dim3 dimGrid(ceil(N / 32.0f), ceil(N/ 16.0f)); // Define grid size
    matrixAdd_B<<<dimGrid, dimBlock>>>(d_a, d_b, d_c,N); // Launch the kernel by specifying grid and block dimensions
    cudaDeviceSynchronize(); // Wait for the GPU to finish execution and check for errors because of the kernel launch

    cudaMemcpy(C,d_c,N*N*sizeof(float),cudaMemcpyDeviceToHost); // Copy the result back to the host
    // Print the result
    printf("Matrix Addition Results:\n");
    printf("C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) { 

            printf("%.2f ",C[i * N + j]); // Prints each element with 2 decimal precision
        }
        printf("\n"); // Adds a newline after each row
    }
    printf("A:\n");
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", A[i * N + j]); // Prints each element with 2 decimal precision
        }
        printf("\n"); // Adds a newline after each row
    }
     printf("B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", B[i * N + j]); // Prints each element with 2 decimal precision
        }
        printf("\n"); // Adds a newline after each row
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}