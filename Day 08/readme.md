# FlashAttention CUDA Implementation

This repository provides a minimal CUDA C++ implementation of the FlashAttention algorithm for efficient attention computation in transformer models. The code demonstrates block-wise processing of the Query, Key, and Value matrices to optimize memory usage and computational speed on the GPU.

## Features

- **Block-wise attention computation** for improved memory efficiency.
- **Utilizes CUDA shared memory** to accelerate matrix operations.
- **Includes softmax normalization** for attention weights.
- **Simple, readable structure** for educational and experimental use.

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler (e.g., `g++`)

### Building the Code

1. Clone this repository or copy the code files.
2. Compile the code using `nvcc`:

    ```bash
    nvcc -o flash_attention flash_attention.cu
    ```

### Running the Program

Run the compiled binary:

```bash
./flash_attention
```

The program will:

- Initialize random Query, Key, and Value matrices.
- Compute the attention output using the FlashAttention CUDA kernel.
- Print the Query, Key, Value, and Output matrices.

## Code Structure

| File                | Description                                      |
|---------------------|--------------------------------------------------|
| `flash_attention.cu`| Main CUDA C++ source file with kernel and host code |

## Key Parameters

- **SRAM_SIZE**: Size of on-chip shared memory used for block processing (in floats).
- **sequence_length**: Number of tokens in the input sequence.
- **embed_dimension**: Dimensionality of each token's embedding vector.

You can adjust these constants at the top of the source file to experiment with different sequence lengths and embedding sizes.

## How It Works

The code divides the attention computation into blocks that fit into shared memory (`SRAM_SIZE`).

Each CUDA thread processes a row (token) in a block.

For each block:

1. Loads blocks of Key and Value into shared memory.
2. Loads blocks of Query into shared memory.
3. Computes scaled dot-product attention scores.
4. Applies softmax normalization.
5. Computes the weighted sum of Value vectors to produce the output.

## Output Example

The program prints the initialized Query, Key, and Value matrices, followed by the computed Output matrix (the result of the attention operation).

## License

This project is provided for educational and research purposes. See [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by the FlashAttention paper and related open-source implementations.

## Notes

- This implementation is simplified for clarity and learning. For production use or larger models, further optimizations and error handling are recommended.
- The default settings use very small matrices for demonstration. Increase `sequence_length` and `embed_dimension` for more realistic experiments, but ensure your GPU has enough memory.
- Feel free to open issues or submit pull requests for improvements!