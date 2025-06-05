# Day 07: 1D Convolution with Tiling in CUDA

This implementation demonstrates a 1D convolution operation using CUDA with tiling optimization. The code uses shared memory to improve performance by reducing global memory access.

## Implementation Details

### Key Components

1. **Tiling Strategy**
   - Uses shared memory to cache input data
   - Block size of 32 threads
   - Includes halo regions for mask overlap
   - Shared memory size: `32 + Mask_width - 1` elements

2. **Convolution Parameters**
   - Mask width: 5 elements
   - Input array size: 10 elements (configurable)
   - Constant memory used for mask storage

### Memory Management

- **Shared Memory (`S_A`)**: 
  - Caches input data for each block
  - Includes halo regions for mask overlap
  - Size: `32 + Mask_width - 1` elements

- **Constant Memory (`M`)**: 
  - Stores the convolution mask
  - Size: 5 elements (Mask_width)

### Kernel Operation

1. **Data Loading**
   - Main data loaded to shared memory
   - Left halo region loaded for mask overlap
   - Right halo region loaded for mask overlap
   - Thread synchronization after loading

2. **Convolution Computation**
   - Each thread computes one output element
   - Uses shared memory for input data access
   - Applies mask to compute convolution result
   - Handles boundary conditions

## Usage

Compile and run the program:
```bash
nvcc one_d_convolution_with_tiling.cu -o convolution
./convolution
```

## Output

The program prints:
1. Input array (A)
2. Convolution mask (d_m)
3. Result array (C)

## Performance Considerations

- Uses shared memory to reduce global memory access
- Implements tiling for better memory access patterns
- Uses constant memory for mask storage
- Includes proper boundary handling
- Thread synchronization ensures correct shared memory access

## Notes

- The implementation uses a block size of 32 threads
- Mask width is fixed at 5 elements
- Input size is configurable (default: 10 elements)
- Includes error checking for CUDA operations
