# 🚀 CUDA Layer Normalization

This project demonstrates how to implement **Layer Normalization** using CUDA, leveraging **shared memory** and **GPU parallelism** to efficiently normalize each row of a 2D matrix.

---

## 📚 What is Layer Normalization?

Layer Normalization is a technique often used in neural networks to stabilize and accelerate training. It normalizes input features across each row (layer) independently by computing the mean and variance:

\[
\text{mean} = \frac{1}{N} \sum x_i,\quad \text{variance} = \frac{1}{N} \sum (x_i - \text{mean})^2
\]
\[
\text{normalized}_i = \frac{x_i - \text{mean}}{\sqrt{\text{variance} + \epsilon}}
\]

---

## 🧠 Key Concepts

- **Shared Memory**: Temporary, fast-access memory shared across threads in a CUDA block.
- **Dynamic Shared Memory**: Allocated at runtime using `extern __shared__`.
- **Thread Hierarchy**: Each thread processes one row; optional expansion to parallelize columns.
- **CUDA Kernel Launch**: Specified with grid/block dimensions and shared memory size.

---

## 🛠️ Code Highlights

### 🔹 Kernel Function

```cpp
__global__ void LayerNorm(const float* A, float* B, int rows, int cols)

```cpp
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* row_data = shared;

    // Load row into shared memory
    for (int col = threadIdx.y; col < cols; col += blockDim.y) {
        row_data[col] = A[row * cols + col];
    }
    __syncthreads();

    // Compute mean
    float mean = 0.0f;
    for (int col = 0; col < cols; ++col) {
        mean += row_data[col];
    }
    mean /= cols;

    // Compute variance
    float var = 0.0f;
    for (int col = 0; col < cols; ++col) {
        float diff = row_data[col] - mean;
        var += diff * diff;
    }
    var /= cols;

    // Normalize and write output
    for (int col = threadIdx.y; col < cols; col += blockDim.y) {
        B[row * cols + col] = (row_data[col] - mean) / sqrtf(var + 1e-5f);
    }
}
```

---

### 🔹 Kernel Launch Example

```cpp
int rows = ...;
int cols = ...;
dim3 blockDim(1, 32); // 1 row per block, 32 threads per row (adjust as needed)
dim3 gridDim((rows + blockDim.x - 1) / blockDim.x);
size_t sharedMemSize = cols * sizeof(float);

LayerNorm<<<gridDim, blockDim, sharedMemSize>>>(A, B, rows, cols);
```

---

## 🏃‍♂️ How to Run

1. **Compile:**
   ```bash
   nvcc -o layernorm layernorm.cu
   ```
2. **Run:**
   ```bash
   ./layernorm
   ```

---

## 📄 References

- [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

---