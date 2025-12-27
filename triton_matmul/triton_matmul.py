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

# --- 1. Triton Basic Benchmark 

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
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

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

# --- 2. Triton Autotuned Benchmark

@triton.autotune(
    configs=[
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
):
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
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    
    triton_autotuned_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )

# --- 3. Triton 2D Grid Autotuned 

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 4}),
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
):
    """
    Triton kernel for matrix multiplication with a 2D launch grid.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

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

def benchmark_triton_2d_grid_autotuned(A, B, C, M, N, K):
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    
    triton_matmul_kernel_2d_grid[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )

# --- 4. Triton Grouped Autotuned

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}),
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
):
    """
    Triton kernel for matrix multiplication with 'grouped' tiling (L2 Cache Optimization).
    """
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

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

def benchmark_triton_grouped_autotuned(A, B, C, M, N, K):
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    
    triton_matmul_kernel_grouped[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )

# --- 5. Triton Persistent Kernel

@triton.jit
def triton_matmul_kernel_persistent(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    TILE_GROUP_SIZE: tl.constexpr
):
    """
    Optimization: Persistent Threading.
    The Grid size is fixed based on hardware (NUM_SMS).
    Threads loop inside the kernel to process all tiles.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    
    # Each program instance steps by the grid size to process more tiles
    for pid in range(start_pid, total_tiles, NUM_SMS):
        
        # Swizzled Logic mixed with persistent loop
        num_pid_in_group = TILE_GROUP_SIZE * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * TILE_GROUP_SIZE
        group_size_m = min(num_pid_m - first_pid_m, TILE_GROUP_SIZE)
        
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

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

def benchmark_triton_persistent(A, B, C, M, N, K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32):
    # Detect SM count for persistency
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    
    # Grid is fixed to number of SMs (or a multiple)
    grid = (num_sms, )
    
    triton_matmul_kernel_persistent[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SMS=num_sms,
        TILE_GROUP_SIZE=8
    )

# --- Main Runner ---

def main():
    parser = argparse.ArgumentParser(description="Triton GPU Matrix Multiplication Benchmark")
    parser.add_argument("--benchmark", type=str, required=True, 
                        choices=[
                            "triton_matmul_basic", 
                            "triton_matmul_autotuned",
                            "triton_2d_grid_autotuned",
                            "triton_grouped_autotuned",
                            "triton_matmul_persistent"
                        ],
                        help="Which matmul implementation to run.")
    parser.add_argument("--block-size-m", type=int, default=128, help="BLOCK_SIZE_M")
    parser.add_argument("--block-size-n", type=int, default=128, help="BLOCK_SIZE_N")
    parser.add_argument("--block-size-k", type=int, default=64, help="BLOCK_SIZE_K")
    parser.add_argument("--N", type=int, required=True, help="Matrix size (N x N)")
    parser.add_argument("--reps", type=int, default=50, help="Number of repetitions")
    parser.add_argument("--mode", type=str, default="multi_run_timing", choices=["multi_run_timing"],
                        help="Only 'multi_run_timing' mode is supported.")

    args = parser.parse_args()

    print(f"Running benchmark: {args.benchmark} with N={args.N}, reps={args.reps}")

    # Initialize data on the GPU
    DTYPE = torch.float32
    M, N, K = args.N, args.N, args.N
    A = torch.randn((M, K), device='cuda', dtype=DTYPE)
    B = torch.randn((K, N), device='cuda', dtype=DTYPE)
    C = torch.zeros((M, N), device='cuda', dtype=DTYPE)

    # --- Select function to run ---
    run_fn = None
    if args.benchmark == "triton_matmul_basic":
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
    elif args.benchmark == "triton_matmul_persistent":
        run_fn = lambda: benchmark_triton_persistent(
            A, B, C, M, N, K,
            BLOCK_SIZE_M=args.block_size_m,
            BLOCK_SIZE_N=args.block_size_n,
            BLOCK_SIZE_K=args.block_size_k
        )
    else:
        print(f"Error: Unknown benchmark '{args.benchmark}'", file=sys.stderr)
        sys.exit(1)

    # --- Warm-up, JIT Compilation, and Verification ---
    print("Warming up, compiling kernels, and verifying results...")
    
    C_ref = torch.matmul(A, B)
    C.zero_()
    run_fn()
    
    try:
        assert torch.allclose(C, C_ref, atol=1e-2, rtol=1e-4)
        print("Verification successful.")
    except AssertionError as e:
        print(f"VERIFICATION FAILED for {args.benchmark}: {e}", file=sys.stderr)

    # --- Execute based on mode ---
    if args.mode == "multi_run_timing":
        timings = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        print(f"Running benchmark {args.reps} times...")
        for _ in range(args.reps):
            torch.cuda.synchronize()
            start_event.record()
            run_fn()
            end_event.record()
            end_event.synchronize()
            
            time_ms = start_event.elapsed_time(end_event)
            timings.append(time_ms / 1000.0)
            
        if timings:
            avg_time = float(np.mean(timings))
        else:
            avg_time = float("nan")

        print(f"avg_time_sec={avg_time:.9f}")

if __name__ == "__main__":
    main()