import numpy as np
import numba
from numba import jit, prange
import time
import math
import argparse
import sys

# --- Matrix Addition Implementations ---

@jit(nopython=True, cache=True)
def numba_naive_add(A, B, C):
    """Numba-jitted naive matrix addition."""
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            C[i, j] = A[i, j] + B[i, j]
    return C

@jit(nopython=True, cache=True)
def numba_tiled_add(A, B, C, B1):
    """Numba-jitted 1-level tiled matrix addition."""
    N = A.shape[0]
    for i_tile in range(0, N, B1):
        for j_tile in range(0, N, B1):
            for i in range(i_tile, min(i_tile + B1, N)):
                for j in range(j_tile, min(j_tile + B1, N)):
                    C[i, j] = A[i, j] + B[i, j]
    return C

@jit(nopython=True, cache=True)
def numba_tiled2_add(A, B, C, B1_L2, B2_L1):
    """Numba-jitted 2-level tiled matrix addition."""
    N = A.shape[0]
    for i2 in range(0, N, B1_L2):
        for j2 in range(0, N, B1_L2):
            for i1 in range(i2, min(i2 + B1_L2, N), B2_L1):
                for j1 in range(j2, min(j2 + B1_L2, N), B2_L1):
                    for i in range(i1, min(i1 + B2_L1, N)):
                        for j in range(j1, min(j1 + B2_L1, N)):
                            C[i, j] = A[i, j] + B[i, j]
    return C

# --- Parallel Matrix Addition ---

@jit(nopython=True, cache=True, parallel=True)
def numba_parallel_naive_add(A, B, C):
    """Numba-jitted naive matrix addition, parallelized over rows."""
    N = A.shape[0]
    for i in prange(N): # Parallelized outer loop
        for j in range(N):
            C[i, j] = A[i, j] + B[i, j]
    return C

@jit(nopython=True, cache=True, parallel=True)
def numba_parallel_tiled_add(A, B, C, B1):
    """Numba-jitted 1-level tiled matrix addition, parallelized over tiles."""
    N = A.shape[0]
    
    # FIX: Calculate number of tiles and parallelize over tile *index*
    # (N + B1 - 1) // B1 is a way to do math.ceil(N / B1)
    num_tiles_i = (N + B1 - 1) // B1
    num_tiles_j = (N + B1 - 1) // B1
    
    # prange now has a constant step of 1
    for tile_idx_i in prange(num_tiles_i):
        i_tile = tile_idx_i * B1 # Reconstruct tile start
        
        for tile_idx_j in range(num_tiles_j): # Inner loop is serial
            j_tile = tile_idx_j * B1
            
            for i in range(i_tile, min(i_tile + B1, N)):
                for j in range(j_tile, min(j_tile + B1, N)):
                    C[i, j] = A[i, j] + B[i, j]
    return C

# --- Matrix Multiplication Implementations (Standard) ---

@jit(nopython=True, cache=True)
def numba_naive_mul(A, B, C):
    """Numba-jitted naive matrix multiplication. C must be pre-zeroed."""
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i, k] * B[k, j] 
            C[i, j] = tmp
    return C

@jit(nopython=True, cache=True)
def numba_tiled_mul(A, B, C, B1):
    """Numba-jitted 1-level (6-loop) tiled matrix multiplication. C must be pre-zeroed."""
    N = A.shape[0]
    for ii in range(0, N, B1):
        for jj in range(0, N, B1):
            for kk in range(0, N, B1):
                for i in range(ii, min(ii + B1, N)):
                    for j in range(jj, min(jj + B1, N)):
                        tmp = 0.0
                        for k in range(kk, min(kk + B1, N)):
                            tmp += A[i, k] * B[k, j] 
                        C[i, j] += tmp
    return C

@jit(nopython=True, cache=True)
def numba_tiled2_mul(A, B, C, B1_L2, B2_L1):
    """Numba-jitted 2-level (9-loop) tiled matrix multiplication. C must be pre-zeroed."""
    N = A.shape[0]
    for i2 in range(0, N, B1_L2):
        for j2 in range(0, N, B1_L2):
            for k2 in range(0, N, B1_L2):
                for i1 in range(i2, min(i2 + B1_L2, N), B2_L1):
                    for j1 in range(j2, min(j2 + B1_L2, N), B2_L1):
                        for k1 in range(k2, min(k2 + B1_L2, N), B2_L1):
                            for i in range(i1, min(i1 + B2_L1, N)):
                                for j in range(j1, min(j1 + B2_L1, N)):
                                    tmp = 0.0
                                    for k in range(k1, min(k1 + B2_L1, N)):
                                        tmp += A[i, k] * B[k, j] 
                                    C[i, j] += tmp
    return C

# --- Matrix Multiplication Implementations (Transposed B) ---

@jit(nopython=True, cache=True)
def numba_naive_mul_transposed(A, B_T, C):
    """Numba-jitted naive matmul using B_Transposed. C must be pre-zeroed."""
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i, k] * B_T[j, k] # Contiguous access to B_T
            C[i, j] = tmp
    return C

@jit(nopython=True, cache=True)
def numba_tiled_mul_transposed(A, B_T, C, B1):
    """Numba-jitted 1-level tiled matmul using B_Transposed. C must be pre-zeroed."""
    N = A.shape[0]
    for ii in range(0, N, B1):
        for jj in range(0, N, B1):
            for kk in range(0, N, B1):
                for i in range(ii, min(ii + B1, N)):
                    for j in range(jj, min(jj + B1, N)):
                        tmp = 0.0
                        for k in range(kk, min(kk + B1, N)):
                            tmp += A[i, k] * B_T[j, k] # Contiguous access
                        C[i, j] += tmp
    return C

@jit(nopython=True, cache=True)
def numba_tiled2_mul_transposed(A, B_T, C, B1_L2, B2_L1):
    """Numba-jitted 2-level tiled matmul using B_Transposed. C must be pre-zeroed."""
    N = A.shape[0]
    for i2 in range(0, N, B1_L2):
        for j2 in range(0, N, B1_L2):
            for k2 in range(0, N, B1_L2):
                for i1 in range(i2, min(i2 + B1_L2, N), B2_L1):
                    for j1 in range(j2, min(j2 + B1_L2, N), B2_L1):
                        for k1 in range(k2, min(k2 + B1_L2, N), B2_L1):
                            for i in range(i1, min(i1 + B2_L1, N)):
                                for j in range(j1, min(j1 + B2_L1, N)):
                                    tmp = 0.0
                                    for k in range(k1, min(k1 + B2_L1, N)):
                                        tmp += A[i, k] * B_T[j, k] # Contiguous
                                    C[i, j] += tmp
    return C

# --- Parallel Matrix Multiplication (Standard) ---

@jit(nopython=True, cache=True, parallel=True)
def numba_parallel_naive_mul(A, B, C):
    """Numba-jitted naive matmul, parallelized over rows."""
    N = A.shape[0]
    for i in prange(N): # Parallelized outer loop
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp
    return C

@jit(nopython=True, cache=True, parallel=True)
def numba_parallel_tiled_mul(A, B, C, B1):
    """Numba-jitted 1-level tiled matmul, parallelized over tiles."""
    N = A.shape[0]
    num_tiles = (N + B1 - 1) // B1
    
    for tile_idx_ii in prange(num_tiles): # Parallelize over ii tile index
        ii = tile_idx_ii * B1
        
        for tile_idx_jj in range(num_tiles): # Serial
            jj = tile_idx_jj * B1
            
            for tile_idx_kk in range(num_tiles): # Serial
                kk = tile_idx_kk * B1
                
                for i in range(ii, min(ii + B1, N)):
                    for j in range(jj, min(jj + B1, N)):
                        tmp = 0.0
                        for k in range(kk, min(kk + B1, N)):
                            tmp += A[i, k] * B[k, j]
                        C[i, j] += tmp
    return C

@jit(nopython=True, cache=True, parallel=True)
def numba_parallel_tiled2_mul(A, B, C, B1_L2, B2_L1):
    """Numba-jitted 2-level tiled matmul, parallelized over tiles."""
    N = A.shape[0]
    num_tiles_L2 = (N + B1_L2 - 1) // B1_L2
    
    for tile_idx_i2 in prange(num_tiles_L2): # Parallelize over i2 tile index
        i2 = tile_idx_i2 * B1_L2
        
        for tile_idx_j2 in range(num_tiles_L2): # Serial
            j2 = tile_idx_j2 * B1_L2
            
            for tile_idx_k2 in range(num_tiles_L2): # Serial
                k2 = tile_idx_k2 * B1_L2
                
                for i1 in range(i2, min(i2 + B1_L2, N), B2_L1):
                    for j1 in range(j2, min(j2 + B1_L2, N), B2_L1):
                        for k1 in range(k2, min(k2 + B1_L2, N), B2_L1):
                            for i in range(i1, min(i1 + B2_L1, N)):
                                for j in range(j1, min(j1 + B2_L1, N)):
                                    tmp = 0.0
                                    for k in range(k1, min(k1 + B2_L1, N)):
                                        tmp += A[i, k] * B[k, j]
                                    C[i, j] += tmp
    return C
    
# --- Parallel Matrix Multiplication (Transposed B) ---

@jit(nopython=True, cache=True, parallel=True)
def numba_parallel_naive_mul_transposed(A, B_T, C):
    """Numba-jitted naive matmul (B_T), parallelized over rows."""
    N = A.shape[0]
    for i in prange(N): # Parallelized outer loop
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i, k] * B_T[j, k]
            C[i, j] = tmp
    return C

@jit(nopython=True, cache=True, parallel=True)
def numba_parallel_tiled_mul_transposed(A, B_T, C, B1):
    """Numba-jitted 1-level tiled matmul (B_T), parallelized over tiles."""
    N = A.shape[0]
    num_tiles = (N + B1 - 1) // B1
    
    for tile_idx_ii in prange(num_tiles): # Parallelize over ii tile index
        ii = tile_idx_ii * B1
        
        for tile_idx_jj in range(num_tiles): # Serial
            jj = tile_idx_jj * B1
            
            for tile_idx_kk in range(num_tiles): # Serial
                kk = tile_idx_kk * B1
                
                for i in range(ii, min(ii + B1, N)):
                    for j in range(jj, min(jj + B1, N)):
                        tmp = 0.0
                        for k in range(kk, min(kk + B1, N)):
                            tmp += A[i, k] * B_T[j, k]
                        C[i, j] += tmp
    return C

@jit(nopython=True, cache=True, parallel=True)
def numba_parallel_tiled2_mul_transposed(A, B_T, C, B1_L2, B2_L1):
    """Numba-jitted 2-level tiled matmul (B_T), parallelized over tiles."""
    N = A.shape[0]
    num_tiles_L2 = (N + B1_L2 - 1) // B1_L2
    
    for tile_idx_i2 in prange(num_tiles_L2): # Parallelize over i2 tile index
        i2 = tile_idx_i2 * B1_L2
        
        for tile_idx_j2 in range(num_tiles_L2): # Serial
            j2 = tile_idx_j2 * B1_L2
            
            for tile_idx_k2 in range(num_tiles_L2): # Serial
                k2 = tile_idx_k2 * B1_L2
                
                for i1 in range(i2, min(i2 + B1_L2, N), B2_L1):
                    for j1 in range(j2, min(j2 + B1_L2, N), B2_L1):
                        for k1 in range(k2, min(k2 + B1_L2, N), B2_L1):
                            for i in range(i1, min(i1 + B2_L1, N)):
                                for j in range(j1, min(j1 + B2_L1, N)):
                                    tmp = 0.0
                                    for k in range(k1, min(k1 + B2_L1, N)):
                                        tmp += A[i, k] * B_T[j, k]
                                    C[i, j] += tmp
    return C

# --- Main Runner ---

def main():
    parser = argparse.ArgumentParser(description="Matrix Benchmark Runner")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark function to run")
    parser.add_argument("--N", type=int, required=True, help="Matrix size")
    parser.add_argument("--B1", type=int, default=0, help="Block size 1 (B_L2)")
    parser.add_argument("--B2", type=int, default=0, help="Block size 2 (B_L1)")
    parser.add_argument("--reps", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--mode", type=str, required=True, choices=["single_run", "multi_run_perf", "multi_run_timing"])
    
    args = parser.parse_args()

    # Initialize data
    A = np.random.rand(args.N, args.N).astype(np.float64)
    B = np.random.rand(args.N, args.N).astype(np.float64)
    # Pre-calculate B_T. .copy() is essential for contiguous memory.
    B_T = B.T.copy() 
    C = np.zeros((args.N, args.N), dtype=np.float64)

    # --- Select function to run ---
    run_fn = None
    is_mul = False
    
    # --- Addition ---
    if args.benchmark == "numpy_add":
        run_fn = lambda: (A + B)
    elif args.benchmark == "numba_naive_add":
        run_fn = lambda: numba_naive_add(A, B, C)
    elif args.benchmark == "numba_tiled_add":
        run_fn = lambda: numba_tiled_add(A, B, C, args.B1)
    elif args.benchmark == "numba_tiled2_add":
        run_fn = lambda: numba_tiled2_add(A, B, C, args.B1, args.B2)
    elif args.benchmark == "numba_parallel_naive_add":
        run_fn = lambda: numba_parallel_naive_add(A, B, C)
    elif args.benchmark == "numba_parallel_tiled_add":
        run_fn = lambda: numba_parallel_tiled_add(A, B, C, args.B1)
        
    # --- Multiplication (Standard) ---
    elif args.benchmark == "numpy_mul":
        run_fn = lambda: (A @ B)
        is_mul = True
    elif args.benchmark == "numba_naive_mul":
        run_fn = lambda: numba_naive_mul(A, B, C)
        is_mul = True
    elif args.benchmark == "numba_tiled_mul":
        run_fn = lambda: numba_tiled_mul(A, B, C, args.B1)
        is_mul = True
    elif args.benchmark == "numba_tiled2_mul":
        run_fn = lambda: numba_tiled2_mul(A, B, C, args.B1, args.B2)
        is_mul = True
        
    # --- Multiplication (Transposed) ---
    elif args.benchmark == "numba_naive_mul_transposed":
        run_fn = lambda: numba_naive_mul_transposed(A, B_T, C)
        is_mul = True
    elif args.benchmark == "numba_tiled_mul_transposed":
        run_fn = lambda: numba_tiled_mul_transposed(A, B_T, C, args.B1)
        is_mul = True
    elif args.benchmark == "numba_tiled2_mul_transposed":
        run_fn = lambda: numba_tiled2_mul_transposed(A, B_T, C, args.B1, args.B2)
        is_mul = True
        
    # --- Multiplication (Parallel) ---
    elif args.benchmark == "numba_parallel_naive_mul":
        run_fn = lambda: numba_parallel_naive_mul(A, B, C)
        is_mul = True
    elif args.benchmark == "numba_parallel_tiled_mul":
        run_fn = lambda: numba_parallel_tiled_mul(A, B, C, args.B1)
        is_mul = True
    elif args.benchmark == "numba_parallel_tiled2_mul":
        run_fn = lambda: numba_parallel_tiled2_mul(A, B, C, args.B1, args.B2)
        is_mul = True
        
    # --- Multiplication (Parallel + Transposed) ---
    elif args.benchmark == "numba_parallel_naive_mul_transposed":
        run_fn = lambda: numba_parallel_naive_mul_transposed(A, B_T, C)
        is_mul = True
    elif args.benchmark == "numba_parallel_tiled_mul_transposed":
        run_fn = lambda: numba_parallel_tiled_mul_transposed(A, B_T, C, args.B1)
        is_mul = True
    elif args.benchmark == "numba_parallel_tiled2_mul_transposed":
        run_fn = lambda: numba_parallel_tiled2_mul_transposed(A, B_T, C, args.B1, args.B2)
        is_mul = True
        
    else:
        print(f"Error: Unknown benchmark '{args.benchmark}'", file=sys.stderr)
        sys.exit(1)

    # --- JIT Compilation ---
    # Run once to ensure Numba has compiled everything before timing.
    # This is crucial for 'multi_run_perf' mode.
    if "numba" in args.benchmark:
        run_fn()

    # --- Execute based on mode ---
    
    if args.mode == "multi_run_perf":
        # This mode is for perf to run over.
        # We just run the function N times.
        for _ in range(args.reps):
            if is_mul: C.fill(0.0)
            run_fn()
            
    elif args.mode == "multi_run_timing":
        # This mode runs its own timer and prints CSV
        op_type = "mul" if is_mul else "add"
        
        for i in range(args.reps):
            if is_mul: C.fill(0.0)
            
            start = time.perf_counter()
            run_fn()
            end = time.perf_counter()
            
            # Print CSV row to stdout
            # Header is printed by the shell script
            print(f"{args.benchmark},{op_type},{args.N},{args.B1},{args.B2},{i+1},{end - start:.9f}")

if __name__ == "__main__":
    main()

