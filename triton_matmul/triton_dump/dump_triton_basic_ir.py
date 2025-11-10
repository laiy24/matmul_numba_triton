import numpy as np
import torch
import argparse
import sys
import triton
import triton.language as tl
import os  # <-- Added for file/directory operations

# Check if Triton is available
if triton is None:
    print("Triton not installed. This script requires Triton.", file=sys.stderr)
    sys.exit(1)

# Check for CUDA
if not torch.cuda.is_available():
    print("Error: PyTorch CUDA is not available. This script requires a GPU for compilation context.", file=sys.stderr)
    sys.exit(1)

# --- New Helper Function ---
def save_artifact(content: str, filename: str, output_dir: str = "triton_artifacts"):
    """
    Saves string content to a file in a specified directory.
    The directory will be created if it doesn't exist.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}", file=sys.stderr)
            return

    # Construct the full file path
    filepath = os.path.join(output_dir, filename)

    # Write the content to the file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully saved artifact: {filepath}")
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while saving {filepath}: {e}", file=sys.stderr)

# --- Triton Basic Benchmark Kernel ---
# (This section is unchanged)
@triton.jit
def triton_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    """Triton kernel for matrix multiplication."""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first blocks
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next blocks of A and B
        k_mask_a = (k * BLOCK_SIZE_K + offs_k[None, :]) < K
        k_mask_b = (k * BLOCK_SIZE_K + offs_k[:, None]) < K
        
        a = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)
        
        # Compute matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write the result to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    m_mask = offs_cm[:, None] < M
    n_mask = offs_cn[None, :] < N
    c_mask = m_mask & n_mask
    tl.store(c_ptrs, accumulator, mask=c_mask)

# --- Main function to JIT and dump IR ---

def main():
    """
    Triggers JIT compilation of 'triton_matmul_kernel' by calling it once
    with dummy data, then retrieves all compilation artifacts (TTIR, LLIR, etc.)
    and saves them to files.
    """
    
    # --- 1. Define concrete values to trigger compilation ---
    M, N, K = 512, 512, 512
    DTYPE = torch.float32
    
    # Define constexpr block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # --- Define base name for output files ---
    # This makes filenames descriptive and unique to the compilation parameters
    FILE_PREFIX = f"matmul_M{BLOCK_SIZE_M}_N{BLOCK_SIZE_N}_K{BLOCK_SIZE_K}"
    OUTPUT_DIR = "triton_artifacts"
    
    print(f"Triggering JIT compilation with constants:")
    print(f"  BLOCK_SIZE_M = {BLOCK_SIZE_M}")
    print(f"  BLOCK_SIZE_N = {BLOCK_SIZE_N}")
    print(f"  BLOCK_SIZE_K = {BLOCK_SIZE_K}")
    print(f"Artifacts will be saved to: '{OUTPUT_DIR}/'")
    print("-" * 40)

    # Create dummy tensors on GPU to get valid strides and types
    A = torch.empty((M, K), device='cuda', dtype=DTYPE)
    B = torch.empty((K, N), device='cuda', dtype=DTYPE)
    C = torch.empty((M, N), device='cuda', dtype=DTYPE)

    # Define the grid function
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    # --- 2. Call the kernel to trigger JIT compilation ---
    try:
        triton_matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        torch.cuda.synchronize()

    except Exception as e:
        print(f"An error occurred during the JIT warm-up call:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    # --- 3. Retrieve the compiled object from the cache ---
    # (This cache retrieval logic is unchanged as it seems to work for your version)
    try:
        cache_entry = None
        current_device = torch.cuda.current_device()

        if hasattr(triton_matmul_kernel, 'device_caches'):
            cache_entry = triton_matmul_kernel.device_caches.get(current_device)
        
        if not cache_entry:
            print(f"Error: Found 'device_caches' but it's empty or has no entry for device {current_device}.", file=sys.stderr)
            # ... (debug printing from original script)
            sys.exit(1)

        compiled_object = None
        if isinstance(cache_entry, (tuple, list)) and len(cache_entry) > 0:
            kernel_dict = cache_entry[0]
            if isinstance(kernel_dict, dict) and kernel_dict:
                compiled_object = list(kernel_dict.values())[0]
        elif isinstance(cache_entry, dict):
            if cache_entry:
                compiled_object = list(cache_entry.values())[0]

        if compiled_object is None:
            print(f"Error: Found cache entry for device {current_device}, but could not find a compiled kernel object inside.", file=sys.stderr)
            sys.exit(1)
        
        # --- 4. Dump all available artifacts to files ---
        
        if not hasattr(compiled_object, 'asm') or not isinstance(compiled_object.asm, dict):
            print("Error: 'compiled_object.asm' attribute not found or is not a dictionary.", file=sys.stderr)
            sys.exit(1)

        print(f"\n" + "=" * 20 + f" Saving All Artifacts " + "=" * 20)
        
        # Iterate through all available artifacts in the 'asm' dictionary
        # This includes 'ttir', 'ttgir', 'llir', 'ptx', and potentially 'cubin'
        saved_artifacts = []
        for artifact_name, content in compiled_object.asm.items():
            if content and isinstance(content, str): # Ensure content is a non-empty string
                # Use the artifact_name (e.g., 'ttir', 'llir') as the file extension
                filename = f"{FILE_PREFIX}.{artifact_name}"
                save_artifact(content, filename, output_dir=OUTPUT_DIR)
                saved_artifacts.append(filename)
            elif content and isinstance(content, bytes):
                # Handle binary artifacts like 'cubin'
                filename = f"{FILE_PREFIX}.{artifact_name}"
                # We need a different save function for bytes
                filepath = os.path.join(OUTPUT_DIR, filename)
                try:
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    print(f"Successfully saved artifact: {filepath}")
                    saved_artifacts.append(filename)
                except Exception as e:
                    print(f"Error saving binary artifact {filepath}: {e}", file=sys.stderr)
            else:
                print(f"Skipping empty or non-string/bytes artifact: {artifact_name}")
        
    except Exception as e:
        print(f"An error occurred during IR retrieval from cache:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()