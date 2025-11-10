import numpy as np
import numba
from numba import jit, prange
import time
import math
import argparse
import sys

@jit(nopython=True)
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

# --- Main Runner ---

def main():
    N = 1024  # Default size
    # Initialize data
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    # Pre-calculate B_T. .copy() is essential for contiguous memory.
    C = np.zeros((N, N), dtype=np.float64)

    numba_tiled_mul(A, B, C, 32)

if __name__ == "__main__":
    main()

