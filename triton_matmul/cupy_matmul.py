import numpy as np
import cupy
import argparse
import sys

# --- 1. CuPy Benchmark (cuBLAS) ---

def benchmark_cupy(A, B, C):
    """Benchmark using CuPy's matmul, which typically wraps cuBLAS."""
    # Use C[:] to write into the existing buffer, avoiding new allocation
    C[:] = cupy.matmul(A, B)

# --- Main Runner ---

def main():
    parser = argparse.ArgumentParser(description="CuPy GPU Matrix Multiplication Benchmark")
    parser.add_argument("--benchmark", type=str, required=True, 
                        choices=["cupy_matmul"])
    parser.add_argument("--N", type=int, required=True, help="Matrix size (N x N)")
    parser.add_argument("--reps", type=int, default=50, help="Number of repetitions")
    parser.add_argument("--mode", type=str, required=True, choices=["multi_run_timing"])

    args = parser.parse_args()

    # Check for unsupported modes from the CPU example
    if args.mode != "multi_run_timing":
        print(f"Error: Only 'multi_run_timing' mode is supported.", file=sys.stderr)
        sys.exit(1)

    print(f"Running benchmark: {args.benchmark} with N={args.N}, reps={args.reps}")

    # Initialize data on the GPU
    # Use float32 for better GPU performance
    DTYPE = cupy.float32
    M, N, K = args.N, args.N, args.N
    A = cupy.random.rand(M, K, dtype=DTYPE)
    B = cupy.random.rand(K, N, dtype=DTYPE)
    C = cupy.zeros((M, N), dtype=DTYPE)

    # --- Select function to run ---
    run_fn = None
    if args.benchmark == "cupy_matmul":
        run_fn = lambda: benchmark_cupy(A, B, C)
    else:
        # This should not be reachable due to argparse choices
        print(f"Error: Unknown benchmark '{args.benchmark}'", file=sys.stderr)
        sys.exit(1)

    # --- Warm-up, JIT Compilation, and Verification ---
    print("Warming up, compiling kernels, and verifying results...")
    
    # Calculate reference result using CuPy
    C_ref = cupy.matmul(A, B)
    
    # Run the selected function once to compile/autotune
    # We need to reset C for verification
    C.fill(0.0)
    run_fn()
    
    # Verify the result
    try:
        cupy.testing.assert_allclose(C, C_ref, rtol=1e-4, atol=1e-4)
        print("Verification successful.")
    except Exception as e:
        print(f"VERIFICATION FAILED for {args.benchmark}: {e}", file=sys.stderr)
        # Optionally, exit if verification fails
        # sys.exit(1)

    # --- Execute based on mode ---
    if args.mode == "multi_run_timing":
        timings = []
        
        # Create CUDA events for accurate timing
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()
        
        print(f"Running benchmark {args.reps} times...")
        for _ in range(args.reps):
            # C.fill(0.0) # Reset C (matmul overwrites, so not strictly needed)
            
            # Ensure all previous GPU work is done
            cupy.cuda.runtime.deviceSynchronize()
            
            # Record start event
            start_event.record()
            
            # Run the function
            run_fn()
            
            # Record end event
            end_event.record()
            
            # Wait for the end event to complete
            end_event.synchronize()
            
            # Get elapsed time in milliseconds and convert to seconds
            time_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
            timings.append(time_ms / 1000.0)
            
        if timings:
            avg_time = float(np.mean(timings))
        else:
            avg_time = float("nan")

        # Emit the single summary line
        print(f"avg_time_sec={avg_time:.9f}")

if __name__ == "__main__":
    main()