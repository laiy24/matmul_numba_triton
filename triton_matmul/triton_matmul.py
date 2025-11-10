import numpy as np
import torch
import argparse
import sys
import triton
import triton.language as tl

# Check if Triton is available
if triton is None:
    print("Triton not installed. This script requires Triton.", file=sys.stderr)
    sys.exit(1)

# Check for CUDA
if not torch.cuda.is_available():
    print("Error: PyTorch CUDA is not available. This benchmark requires a GPU.", file=sys.stderr)
    sys.exit(1)

# --- 1. Triton Basic Benchmark ---

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

def benchmark_triton_basic(A, B, C, M, N, K):
    """Launcher for the basic Triton kernel with fixed block sizes."""
    
    # Grid definition
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    # Launch kernel
    # BLOCK sizes are provided by the caller (to allow sweeping from the CLI)
    def launcher(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32):
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

    return launcher

# --- 2. Triton Autotuned Benchmark (Original) ---

@triton.autotune(
    configs=[
        # --- EXPANDED COMPREHENSIVE CONFIG LIST ---
        # "Basic" config + variations
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 3}),
        
        # Small blocks + variations
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_warps': 2, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 2, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_warps': 2, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),

        # Balanced blocks + variations
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 5}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 5}),

        # Tall blocks (M-heavy) + variations
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 16, 'num_stages': 4}),

        # Wide blocks (N-heavy) + variations
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 16, 'num_stages': 4}),

        # Large blocks + variations
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 16, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 16, 'num_stages': 3}),

        # Very large blocks + variations
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 16, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 32, 'num_warps': 16, 'num_stages': 2}),
        # --- END OF LIST ---
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_autotuned_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
    # num_warps and num_stages are used by the autotuner
):
    # This kernel is identical to the basic one,
    # the autotuner finds the best BLOCK sizes and compiler params.
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask_a = (k * BLOCK_SIZE_K + offs_k[None, :]) < K
        k_mask_b = (k * BLOCK_SIZE_K + offs_k[:, None]) < K
        
        a = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)
        
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    m_mask = offs_cm[:, None] < M
    n_mask = offs_cn[None, :] < N
    c_mask = m_mask & n_mask
    tl.store(c_ptrs, accumulator, mask=c_mask)

def benchmark_triton_autotuned(A, B, C, M, N, K):
    """Launcher for the autotuned Triton kernel."""
    
    # Grid definition
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    
    # Autotuner runs here
    triton_autotuned_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )

# --- 3. NEW: Triton 2D Grid Autotuned ---

@triton.autotune(
    configs=[
        # --- EXPANDED COMPREHENSIVE CONFIG LIST ---
        # "Basic" config + variations
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 3}),
        
        # Small blocks + variations
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_warps': 2, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 2, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_warps': 2, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),

        # Balanced blocks + variations
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 5}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 5}),

        # Tall blocks (M-heavy) + variations
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 16, 'num_stages': 4}),

        # Wide blocks (N-heavy) + variations
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 16, 'num_stages': 4}),

        # Large blocks + variations
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 16, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 16, 'num_stages': 3}),

        # Very large blocks + variations
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 16, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 32, 'num_warps': 16, 'num_stages': 2}),
        # --- END OF LIST ---
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_matmul_kernel_2d_grid(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
    # num_warps and num_stages are used by the autotuner
):
    """
    Triton kernel for matrix multiplication with a 2D launch grid.
    This is a more natural mapping for 2D problems.
    """
    # Use 2D program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

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

def benchmark_triton_2d_grid_autotuned(A, B, C, M, N, K):
    """Launcher for the 2D grid autotuned Triton kernel."""
    
    # Grid definition (now 2D)
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    
    # Autotuner runs here
    triton_matmul_kernel_2d_grid[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )

# --- 4. NEW: Triton Grouped Autotuned ---

@triton.autotune(
    configs=[
        # --- EXPANDED COMPREHENSIVE CONFIG LIST (with GROUP_M) ---
        # "Basic" config + variations
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}),
        
        # Balanced + variations
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_M': 4, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_M': 16, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 5}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 5}),

        # Tall + variations
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_M': 4, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_M': 16, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_M': 16, 'num_warps': 16, 'num_stages': 4}),
        
        # Wide + variations
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_M': 4, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_M': 8, 'num_warps': 16, 'num_stages': 4}),

        # Large + variations
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_M': 4, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_M': 8, 'num_warps': 16, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_M': 16, 'num_warps': 16, 'num_stages': 3}),
        # --- END OF LIST ---
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_matmul_kernel_grouped(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M: tl.constexpr
    # num_warps and num_stages are used by the autotuner
):
    """
    Triton kernel for matrix multiplication with 'grouped' tiling.
    Blocks are grouped along the M-dimension to improve L2 cache reuse.
    """
    # 1D launch grid
    pid = tl.program_id(axis=0)
    
    # Grouping logic
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    
    # Get 2D tile indices
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

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

def benchmark_triton_grouped_autotuned(A, B, C, M, N, K):
    """Launcher for the grouped autotuned Triton kernel."""
    
    # Grid definition (1D)
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    
    # Autotuner runs here
    triton_matmul_kernel_grouped[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
        # GROUP_M is handled by the autotuner
    )

# --- Main Runner ---

def main():
    parser = argparse.ArgumentParser(description="Triton GPU Matrix Multiplication Benchmark")
    parser.add_argument("--benchmark", type=str, required=True, 
                        choices=[
                            "triton_matmul_basic", 
                            "triton_matmul_autotuned",
                            "triton_2d_grid_autotuned", # NEW
                            "triton_grouped_autotuned"  # NEW
                        ],
                        help="Which matmul implementation to run.")
    parser.add_argument("--block-size-m", type=int, default=128, help="BLOCK_SIZE_M for basic triton kernel")
    parser.add_argument("--block-size-n", type=int, default=128, help="BLOCK_SIZE_N for basic triton kernel")
    parser.add_argument("--block-size-k", type=int, default=64, help="BLOCK_SIZE_K for basic triton kernel")
    parser.add_argument("--N", type=int, required=True, help="Matrix size (N x N)")
    parser.add_argument("--reps", type=int, default=50, help="Number of repetitions")
    parser.add_argument("--mode", type=str, default="multi_run_timing", choices=["multi_run_timing"],
                        help="Only 'multi_run_timing' mode is supported.")

    args = parser.parse_args()

    print(f"Running benchmark: {args.benchmark} with N={args.N}, reps={args.reps}")

    # Initialize data on the GPU
    # Use float32 for better GPU performance
    DTYPE = torch.float32
    M, N, K = args.N, args.N, args.N
    A = torch.randn((M, K), device='cuda', dtype=DTYPE)
    B = torch.randn((K, N), device='cuda', dtype=DTYPE)
    C = torch.zeros((M, N), device='cuda', dtype=DTYPE)

    # --- Select function to run ---
    run_fn = None
    if args.benchmark == "triton_matmul_basic":
        # benchmark_triton_basic returns a launcher callable that accepts block sizes
        run_fn = lambda: benchmark_triton_basic(A, B, C, M, N, K)(
            BLOCK_SIZE_M=args.block_size_m,
            BLOCK_SIZE_N=args.block_size_n,
            BLOCK_SIZE_K=args.block_size_k
        )
    elif args.benchmark == "triton_matmul_autotuned":
        run_fn = lambda: benchmark_triton_autotuned(A, B, C, M, N, K)
    elif args.benchmark == "triton_2d_grid_autotuned":
        run_fn = lambda: benchmark_triton_2d_grid_autotuned(A, B, C, M, N, K)
    elif args.benchmark == "triton_grouped_autotuned":
        run_fn = lambda: benchmark_triton_grouped_autotuned(A, B, C, M, N, K)
    else:
        # This should not be reachable due to argparse choices
        print(f"Error: Unknown benchmark '{args.benchmark}'", file=sys.stderr)
        sys.exit(1)

    # --- Warm-up, JIT Compilation, and Verification ---
    print("Warming up, compiling kernels, and verifying results...")
    
    # Calculate reference result using PyTorch
    C_ref = torch.matmul(A, B)
    
    # Run the selected function once to compile/autotune
    # We need to reset C for verification
    C.zero_()
    run_fn()
    
    # Verify the result
    try:
        assert torch.allclose(C, C_ref, atol=1e-2, rtol=1e-4)
        print("Verification successful.")
    except AssertionError as e:
        print(f"VERIFICATION FAILED for {args.benchmark}: {e}", file=sys.stderr)
        # Optionally, exit if verification fails
        # sys.exit(1)

    # --- Execute based on mode ---
    if args.mode == "multi_run_timing":
        timings = []
        
        # Create CUDA events for accurate timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        print(f"Running benchmark {args.reps} times...")
        for _ in range(args.reps):
            # C.zero_() # Reset C (matmul overwrites, so not strictly needed)
            
            # Ensure all previous GPU work is done
            torch.cuda.synchronize()
            
            # Record start event
            start_event.record()
            
            # Run the function
            run_fn()
            
            # Record end event
            end_event.record()
            
            # Wait for the end event to complete
            end_event.synchronize()
            
            # Get elapsed time in milliseconds and convert to seconds
            time_ms = start_event.elapsed_time(end_event)
            timings.append(time_ms / 1000.0)
            
        if timings:
            avg_time = float(np.mean(timings))
        else:
            avg_time = float("nan")

        # Emit the single summary line
        print(f"avg_time_sec={avg_time:.9f}")

if __name__ == "__main__":
    main()