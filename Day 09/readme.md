# Brent-Kung Prefix Sum CUDA Implementation

![CUDA](https://img.shields.io/badge/CUDA-11%2B-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains a CUDA implementation of the Brent-Kung parallel prefix sum algorithm, optimized for GPU computation. The code demonstrates efficient parallel scan operations using shared memory and block-wise processing.

## Features

- Brent-Kung Algorithm implementation for parallel prefix sums
- Two-phase processing:
    - **Up-sweep**: Reduction phase building partial sums
    - **Down-sweep**: Distribution phase propagating sums
- Shared memory optimization for fast data access
- Batched processing of 2 elements per thread
- Error checking for CUDA API calls

## Prerequisites

- NVIDIA GPU with CUDA support (Compute Capability 3.5+)
- CUDA Toolkit 11+
- GCC or compatible C++ compiler

## Quick Start

### Compilation
```bash
nvcc -o prefix_sum prefix_sum.cu
```

### Execution
```bash
./prefix_sum
```

## Code Structure

| Component | Description |
|-----------|-------------|
| `prefixsum_kernel` | CUDA kernel implementing Brent-Kung phases |
| `checkCudaError` | Helper function for CUDA error handling |
| `main` | Host code with data initialization & I/O |

## Key Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `LOAD_SIZE` | 32 | Shared memory size per block (floats) |
| `N` | 10 | Default input size (configurable) |

## Example Output

```text
A:
1.00 2.00 3.00 4.00 5.00 6.00 7.00 8.00 9.00 10.00 
C:
1.00 3.00 6.00 10.00 15.00 21.00 28.00 36.00 45.00 55.00 
```

## Performance Notes

- Optimal for power-of-two input sizes
- Current implementation uses 32 threads per block
- Shared memory size (`LOAD_SIZE`) should match GPU capabilities
- For larger datasets, modify `N` and ensure adequate GPU memory

## License

This project is licensed under the MIT License - see LICENSE for details.

## References

- Brent-Kung algorithm original paper
- CUDA C++ Programming Guide
- Parallel prefix sum optimizations

## Contributing

Contributions are welcome! Please open an issue or PR for:

- Performance improvements
- Additional algorithm variants
- Enhanced error handling

This README provides essential information for users to quickly understand and utilize your Brent-Kung implementation. You might want to add installation details for dependencies or usage examples for different input sizes based on your specific needs.