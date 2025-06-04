# CUDA Matrix Transposition

This project demonstrates how to perform **matrix transposition** using NVIDIA CUDA for GPU acceleration. The program creates a 1024x1024 matrix, transposes it using a CUDA kernel, and verifies the result on the host.

## 🚀 Features

- Matrix transposition using GPU parallelism
- CUDA kernel with 2D grid and block configuration
- Error checking for CUDA operations
- Result validation on the host
- Efficient use of device and host memory


## 📦 Requirements

- CUDA Toolkit (10.0 or higher recommended)
- NVIDIA GPU with CUDA support
- C++ Compiler (e.g., `g++`)
- `nvcc` (NVIDIA CUDA compiler)

## 🛠️ Compilation

To compile the code using `nvcc`, run:

```bash
nvcc MatrixTranspose.cu -o transpose
```

## ▶️ Running the Program

Once compiled, execute the binary:

```bash
./transpose
```

You should see:

```
Matrix transposition succeeded!
```

## 🧠 How It Works

1. **Matrix Initialization**: A 1024×1024 matrix is created on the host and filled with sequential values.
2. **Memory Transfer**: The input matrix is copied to the device (GPU).
3. **Kernel Launch**: The transposition is performed by the `transposeMatrix` kernel on the GPU.
4. **Result Retrieval**: The transposed matrix is copied back to the host.
5. **Verification**: The result is compared with the expected output to ensure correctness.

## 📌 Kernel Explanation

```cpp
__global__ void transposeMatrix(const float *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}
```

## ✅ Sample Output

```
Matrix transposition succeeded!
```

If there's any CUDA error, it will be printed to the console with a descriptive message.

## 📃 License

This project is open-source and free to use for educational and research purposes.

---

Made with ❤️ using CUDA.
