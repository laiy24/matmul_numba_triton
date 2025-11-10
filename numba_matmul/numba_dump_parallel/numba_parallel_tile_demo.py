import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def numba_parallel_tiled_mul(A, B, C, B1):
    """Numba-jitted 1-level tiled matrix multiplication with parallel tiles. C must be pre-zeroed."""
    N = A.shape[0]
    num_tiles = (N + B1 - 1) // B1

    for tile_idx_ii in prange(num_tiles):
        ii = tile_idx_ii * B1

        for tile_idx_jj in range(num_tiles):
            jj = tile_idx_jj * B1

            for tile_idx_kk in range(num_tiles):
                kk = tile_idx_kk * B1

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
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    numba_parallel_tiled_mul(A, B, C, 32)

if __name__ == "__main__":
    main()
