#!/bin/bash

echo "--- Starting Matrix Benchmark Automation ---"

# --- Configuration ---
#
# *** EDIT THIS ***
# Paste the full path to the python executable from your conda environment
# Find it by running: `conda activate <your_env>` and then `which python`
PYTHON_CMD="/home/laiy24/miniconda3/envs/numba/bin/python"
#
# ---

# Number of times to repeat *inside* each Python run
# NOTE: 100 reps can take a very long time for parallel N=1024.
# Consider lowering to 10 or 20 if runtime is too long.
REPS=30

# List of matrix sizes to test
N_SIZES="1024 2048"
# Block sizes for L2 (outer tile)
B_SIZES="4 8 16 32 64 128 256" 
# Block sizes for L1 (inner tile)
B2_SIZES="4 8 16 32 64 128"

# Perf events to monitor
# Your system might require different names. If this fails, run `perf list`.
PERF_EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L2-loads,L2-load-misses,LLC-loads,LLC-load-misses"

# Output file
OUT_CSV="numba_benchmark_results.csv"

# --- Setup ---
# Clean previous results
rm -f $OUT_CSV

# Check for perf
if ! command -v perf &> /dev/null
then
    echo "Error: 'perf' command not found. Please install 'linux-tools-common' or equivalent."
    exit 1
fi

# Check for python
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "Error: Python command not found at '$PYTHON_CMD'"
    echo "Please edit 'numba_benchmark.sh' and set the PYTHON_CMD variable."
    exit 1
fi
echo "Using Python command: $PYTHON_CMD"


# --- Helper Function for Perf ---
# $1: Benchmark Name
# $2: N
# $3: B1
# $4: B2
# $5: Reps
# $6: Output CSV File
run_perf_stat() {
    local bench_name=$1
    local N=$2
    local B1=$3
    local B2=$4
    local reps=$5
    local out_file=$6
    local avg_time="NA"
    
    # Mode is hardcoded to "multi_run_perf"
    # The Python script will handle the looping.
    local mode="multi_run_perf"

    echo "    Running PERF (N=$N, $reps reps): $bench_name B1=$B1 B2=$B2"
    
    # Get the raw output from perf (which goes to stderr)
    local RAW_PERF_OUTPUT=$(perf stat -x, -e $PERF_EVENTS \
                    $PYTHON_CMD numba_matmul.py \
                    --benchmark $bench_name \
                    --N $N \
                    --B1 $B1 \
                    --B2 $B2 \
                    --mode $mode \
                    --reps $reps 2>&1)
    
    # Check for perf paranoid error
    if [[ "$RAW_PERF_OUTPUT" == *"limited"* ]]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Error: 'perf' access is limited."
        echo "Your 'kernel.perf_event_paranoid' setting is too high."
        echo "Try running this script with 'sudo ./numba_benchmark.sh'"
        echo "Or run: 'sudo sysctl -w kernel.perf_event_paranoid=1'"
        echo "Aborting."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    
    # --- NEW CHECK for bad perf events ---
    if [[ "$RAW_PERF_OUTPUT" == *"Bad event name"* || "$RAW_PERF_OUTPUT" == *"event syntax error"* || "$RAW_PERF_OUTPUT" == *"not supported"* ]]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Error: 'perf' failed to find an event for '$bench_name' (N=$N)."
        echo "Your PERF_EVENTS list in 'numba_benchmark.sh' is likely incorrect for your CPU."
        echo "Failing event list: $PERF_EVENTS"
        echo "Please run 'perf list' to find valid cache event names for your system."
        echo "Common alternatives for L2:"
        echo "  - L2-loads,L2-load-misses"
        echo "  - l2_rqsts.references,l2_rqsts.misses (Intel)"
        echo "  - L2_cache_access,L2_cache_misses"
        echo "Writing 'NA' values to CSV for this run."
        echo "Raw perf error: $RAW_PERF_OUTPUT"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        
        # Write NA for all 8 perf events
        local PERF_VALUES="NA,NA,NA,NA,NA,NA,NA,NA"
    
    else
        # Process the raw output (no error)
        # 1. Filter out blank lines (grep)
        # 2. For each CSV line, take the first field (the value) (awk)
        # 3. Join all lines of values with a comma (paste)
        local PERF_VALUES=$(echo "$RAW_PERF_OUTPUT" | grep -v "^[ \t]*$" | awk -F, '{ print $1 }' | paste -sd,)
    fi
    
    # --- Run timing mode to capture wall-clock runtime ---
    local TIMING_OUTPUT=$( $PYTHON_CMD numba_matmul.py \
                    --benchmark $bench_name \
                    --N $N \
                    --B1 $B1 \
                    --B2 $B2 \
                    --mode multi_run_timing \
                    --reps $reps 2>&1 )

    if [[ $TIMING_OUTPUT =~ avg_time_sec=([0-9eE\.+-]+) ]]; then
        avg_time=${BASH_REMATCH[1]}
    else
        echo "Warning: Unable to parse timing output for $bench_name (N=$N, B1=$B1, B2=$B2)." >&2
        echo "Raw timing output:" >&2
        echo "$TIMING_OUTPUT" >&2
        avg_time="NA"
    fi

    # Write the final, clean CSV row
    echo "$bench_name,$N,$B1,$B2,$reps,$PERF_VALUES,$avg_time" >> $out_file
}

# ===================================================================
# --- Main Benchmark Loop ---
# ===================================================================
echo -e "\n--- Running 'perf stat' for N in {$N_SIZES}, $REPS times each ---"

# Write the new, correct CSV header
# We add "reps" to know what to divide by
echo "benchmark_name,N,B1,B2,reps,$PERF_EVENTS,avg_time_sec" > $OUT_CSV

for N in $N_SIZES; do
    echo "--- Benchmarking for N=$N ($REPS reps) ---"
    
    # --- Addition (N) ---
    echo "  Benchmarking Addition..."
    run_perf_stat "numpy_add" $N 0 0 $REPS $OUT_CSV
    run_perf_stat "numba_naive_add" $N 0 0 $REPS $OUT_CSV
    run_perf_stat "numba_parallel_naive_add" $N 0 0 $REPS $OUT_CSV
    for B1 in $B_SIZES; do
        # Run 1-level tiling
        run_perf_stat "numba_tiled_add" $N $B1 0 $REPS $OUT_CSV
        run_perf_stat "numba_parallel_tiled_add" $N $B1 0 $REPS $OUT_CSV
        
        # Run 2-level tiling with varying B2
        for B2 in $B2_SIZES; do
            # Only run if B2 (inner) is smaller than B1 (outer)
            if [ $B1 -gt $B2 ]; then
                run_perf_stat "numba_tiled2_add" $N $B1 $B2 $REPS $OUT_CSV
                # Note: Not adding parallel for 2-level add, scope is already large
            fi
        done
    done

    # --- Multiplication (N) ---
    echo "  Benchmarking Multiplication..."
    run_perf_stat "numpy_mul" $N 0 0 $REPS $OUT_CSV
    # Naive
    run_perf_stat "numba_naive_mul" $N 0 0 $REPS $OUT_CSV
    run_perf_stat "numba_naive_mul_transposed" $N 0 0 $REPS $OUT_CSV
    run_perf_stat "numba_parallel_naive_mul" $N 0 0 $REPS $OUT_CSV
    run_perf_stat "numba_parallel_naive_mul_transposed" $N 0 0 $REPS $OUT_CSV

    for B1 in $B_SIZES; do
        # Run 1-level tiling
        run_perf_stat "numba_tiled_mul" $N $B1 0 $REPS $OUT_CSV
        run_perf_stat "numba_tiled_mul_transposed" $N $B1 0 $REPS $OUT_CSV
        run_perf_stat "numba_parallel_tiled_mul" $N $B1 0 $REPS $OUT_CSV
        run_perf_stat "numba_parallel_tiled_mul_transposed" $N $B1 0 $REPS $OUT_CSV
        
        # Run 2-level tiling with varying B2
        for B2 in $B2_SIZES; do
            # Only run if B2 (inner) is smaller than B1 (outer)
            if [ $B1 -gt $B2 ]; then
                run_perf_stat "numba_tiled2_mul" $N $B1 $B2 $REPS $OUT_CSV
                run_perf_stat "numba_tiled2_mul_transposed" $N $B1 $B2 $REPS $OUT_CSV
                run_perf_stat "numba_parallel_tiled2_mul" $N $B1 $B2 $REPS $OUT_CSV
                run_perf_stat "numba_parallel_tiled2_mul_transposed" $N $B1 $B2 $REPS $OUT_CSV
            fi
        done
    done
done


echo -e "\n--- All benchmarks complete. ---"
echo "Results are in:"
echo "  - $OUT_CSV (Detailed, clean CSV for analysis)"
echo "Done."

