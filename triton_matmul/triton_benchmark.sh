#!/bin/bash
#
# This script automatically runs all Triton benchmarks from 
# 'triton_matmul.py' across various matrix sizes 
# and saves the results to a CSV file.
#

set -u

echo "--- Starting Triton GPU Matrix Benchmark Automation ---"

# --- Configuration ---
#
# *** EDIT THIS ***
# Paste the full path to the python executable from your conda environment
# Find it by running: `conda activate <your_env>` and then `which python`
PYTHON_CMD="/home/isabelle/miniconda3/envs/triton_env/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ---

# Print autotune is set to 1
export TRITON_PRINT_AUTOTUNING=1

# Number of times to repeat *inside* each Python run
# GPU ops are fast, so a higher rep count gives more stable timings.
REPS=50

# List of matrix sizes to test (N x N)
N_SIZES="512 1024 2048 4096"

# List of benchmarks to run (must match choices in the python script)
# Note: `triton_matmul_basic2` was removed from the Python script; we use
# `triton_matmul_basic` with explicit block-size flags instead.
# Benchmarks ending with "autotuned" manage their own tuning and will be run once per N.
BENCHMARKS="triton_matmul_basic triton_matmul_autotuned triton_2d_grid_autotuned triton_grouped_autotuned"

# List of BLOCK_SIZE triplets to sweep (format: M,N,K)
# Add or remove tuples here to control the sweep.
BLOCK_SIZES="32,32,64 64,64,32 64,128,64 128,128,64"

# Output file
OUT_CSV="triton_benchmark_results.csv"

# --- CuPy benchmark configuration ----------------------------------------
# Python executable to use for running the CuPy benchmark. Edit if needed.
CUPY_PYTHON_CMD="/home/isabelle/miniconda3/envs/numba/bin/python"
# Path to the CuPy benchmark script (relative to this script dir)
CUPY_SCRIPT="$SCRIPT_DIR/cupy_matmul.py"
# Benchmarks to run in the CuPy script (matches argparse choices in cupy_matmul.py)
CUPY_BENCHMARKS="cupy_matmul"
# -------------------------------------------------------------------------

# Resolve repository-relative paths so the script can be launched from anywhere
BENCHMARK_SCRIPT="$SCRIPT_DIR/triton_matmul.py"

# --- Setup ---
# Clean previous results
rm -f "$OUT_CSV"

# Check for python
if ! command -v "$PYTHON_CMD" &> /dev/null
then
    echo "Error: Python command not found at '$PYTHON_CMD'"
    echo "Please edit 'triton_benchmark.sh' and set the PYTHON_CMD variable."
    exit 1
fi
echo "Using Python command: $PYTHON_CMD"
echo "Saving results to: $OUT_CSV"

# Check for CuPy python & script; if missing we skip CuPy benchmarks
SKIP_CUPY=0
if ! command -v "$CUPY_PYTHON_CMD" &> /dev/null
then
    echo "Warning: CuPy Python command not found at '$CUPY_PYTHON_CMD' - CuPy benchmarks will be skipped"
    SKIP_CUPY=1
else
    echo "Using CuPy Python command: $CUPY_PYTHON_CMD"
fi

if [[ ! -f "$CUPY_SCRIPT" ]]; then
    echo "Warning: CuPy benchmark script not found at '$CUPY_SCRIPT' - CuPy benchmarks will be skipped"
    SKIP_CUPY=1
else
    echo "Found CuPy script: $CUPY_SCRIPT"
fi

# Write the CSV header (include block sizes)
echo "benchmark_name,N,reps,block_size_m,block_size_n,block_size_k,avg_time_sec" > "$OUT_CSV"

# ===================================================================
# --- Main Benchmark Loop ---
# ===================================================================

for N in $N_SIZES; do
    echo "--- Benchmarking for N=$N ($REPS reps) ---"
    
    for BENCH in $BENCHMARKS; do
        if [[ "$BENCH" == *autotuned* ]]; then
            echo "        Running: $BENCH (autotuned, block size handled internally)"

            TIMING_OUTPUT=$("$PYTHON_CMD" "$BENCHMARK_SCRIPT" \
                                --benchmark "$BENCH" \
                                --N "$N" \
                                --reps "$REPS" \
                                --mode multi_run_timing 2>&1)
            status=$?
            best_line=$(echo "$TIMING_OUTPUT" | grep -i -m 1 'best config')
            if [[ -n "$best_line" ]]; then
                echo "$best_line"
            else
                echo "$TIMING_OUTPUT"
            fi

            if [[ $status -ne 0 ]] || [[ "$TIMING_OUTPUT" != *"avg_time_sec="* ]]; then
                echo "                ERROR running $BENCH for N=$N."
                echo "                Output was:"
                echo "$TIMING_OUTPUT"
                avg_time="NA"
            else
                avg_time=$(echo "$TIMING_OUTPUT" | grep 'avg_time_sec=' | cut -d'=' -f2)
            fi

            echo "$BENCH,$N,$REPS,NA,NA,NA,$avg_time" >> "$OUT_CSV"
            continue
        fi

        for BS in $BLOCK_SIZES; do
            IFS=',' read -r BS_M BS_N BS_K <<< "$BS"
            echo "        Running: $BENCH with block sizes ${BS_M},${BS_N},${BS_K}"

            # Run the python script in timing mode with block-size flags
            # We capture stdout which contains our "avg_time_sec=..." line
            # We redirect stderr (2) to stdout (1) to capture any Python errors
            TIMING_OUTPUT=$("$PYTHON_CMD" "$BENCHMARK_SCRIPT" \
                                --benchmark "$BENCH" \
                                --N "$N" \
                                --reps "$REPS" \
                                --mode multi_run_timing \
                                --block-size-m "$BS_M" \
                                --block-size-n "$BS_N" \
                                --block-size-k "$BS_K" 2>&1)
            status=$?
            best_line=$(echo "$TIMING_OUTPUT" | grep -i -m 1 'best config')
            if [[ -n "$best_line" ]]; then
                echo "$best_line"
            else
                echo "$TIMING_OUTPUT"
            fi
        
            # Check for errors in the script
            if [[ $status -ne 0 ]] || [[ "$TIMING_OUTPUT" != *"avg_time_sec="* ]]; then
                echo "                ERROR running $BENCH for N=$N."
                echo "                Output was:"
                echo "$TIMING_OUTPUT"
                avg_time="NA"
            else
                # Parse the output
                avg_time=$(echo "$TIMING_OUTPUT" | grep 'avg_time_sec=' | cut -d'=' -f2)
            fi

            # Write the final, clean CSV row (include block sizes)
            echo "$BENCH,$N,$REPS,$BS_M,$BS_N,$BS_K,$avg_time" >> "$OUT_CSV"
        done
    done
    
    # --- Run CuPy benchmarks for this N (if configured) ---
    if [[ $SKIP_CUPY -eq 0 ]]; then
        for CUPY_BENCH in $CUPY_BENCHMARKS; do
            echo "        Running CuPy: $CUPY_BENCH"

            TIMING_OUTPUT=$("$CUPY_PYTHON_CMD" "$CUPY_SCRIPT" \
                                --benchmark "$CUPY_BENCH" \
                                --N "$N" \
                                --reps "$REPS" \
                                --mode multi_run_timing 2>&1)
            status=$?

            best_line=$(echo "$TIMING_OUTPUT" | grep -i -m 1 'best config')
            if [[ -n "$best_line" ]]; then
                echo "$best_line"
            else
                echo "$TIMING_OUTPUT"
            fi
            if [[ $status -ne 0 ]] || [[ "$TIMING_OUTPUT" != *"avg_time_sec="* ]]; then
                echo "                ERROR running cupy $CUPY_BENCH for N=$N."
                echo "                Output was:"
                echo "$TIMING_OUTPUT"
                avg_time="NA"
            else
                # Parse the output
                avg_time=$(echo "$TIMING_OUTPUT" | grep 'avg_time_sec=' | cut -d'=' -f2)
            fi

            # Write the final, clean CSV row (prefix with cupy_ to distinguish)
            # CuPy doesn't use our block-size flags, so write NA for those fields
            echo "cupy_$CUPY_BENCH,$N,$REPS,NA,NA,NA,$avg_time" >> "$OUT_CSV"
        done
    else
        echo "Skipping CuPy benchmarks for N=$N - not configured or missing dependencies"
    fi

done

echo -e "\n--- All benchmarks complete. ---"
echo "Results are in: $OUT_CSV"
echo "Done."