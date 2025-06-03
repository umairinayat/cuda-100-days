# 🚀 CUDA Partial Inclusive Prefix Sum

This project demonstrates how to compute a **partial inclusive prefix sum (scan)** using CUDA. The computation is parallelized across threads and utilizes **shared memory** for optimized performance.

---

## 📂 File Structure

- `partialSum.cu`: Contains the kernel and host code.
- `README.md`: Explains the implementation, usage, and output.

---

## 📌 Description

This program calculates **partial prefix sums** on the GPU using CUDA. It performs the sum of **pairs of elements** and then computes an **inclusive scan (prefix sum)** for each block using **shared memory**.

This is useful in reducing large datasets in a parallelizable and optimized way before performing further computations.

---

## 🧠 Core Concepts

- CUDA kernel launch  
- Shared memory usage  
- Thread indexing and coalesced memory access  
- Prefix sum (inclusive scan)  

---

## 💻 How It Works

1. Each thread reads two elements from the global input array.
2. The pair is summed and stored into **shared memory**.
3. A **prefix sum** is computed on this partial data using a stride-based approach.
4. Final results are written back to the output array in global memory.

---

## 🛠️ CUDA Kernel Breakdown

```cpp
__global__ void partialSumKernel(int *input, int *output, int n)

Parameters:

1. input: Input array on device
2. output: Output array on device
3. n: Number of elements

Shared Memory: Used to store partial sums temporarily for each block

Thread Logic:
1. Each thread adds **input[index] + input[index + blockDim.x]**

2. Performs prefix sum in shared memory

3. Stores the result to output[index]
```
⚙️ How to Compile & Run
Compile

```bash
nvcc partialSum.cu -o prefix_sum
```
Run
```bash
./prefix_sum