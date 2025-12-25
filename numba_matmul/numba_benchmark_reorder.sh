#!/bin/bash

echo "--- Starting Loop Reordering Matrix Benchmark ---"

# --- Configuration ---
#
# *** EDIT THIS ***
# Paste the full path to the python executable from your conda environment
# Find it by running: `conda activate <your_env>` and then `which python`
PYTHON_CMD="/home/laiy24/miniconda3/envs/numba/bin/python"
#
# ---

# Number of times to repeat *inside* each Python run
REPS=30

# List of matrix sizes to test
N_SIZES="1024 2048"

# Perf events to monitor
# Your system might require different names. If this fails, run `perf list`.
PERF_EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L2-loads,L2-load-misses,LLC-loads,LLC-load-misses"

# Output file
OUT_CSV="numba_benchmark_reorder_results.csv"

# --- Setup ---
# Clean previous results
rm -f $OUT_CSV

# Check for perf
if ! which perf > /dev/null 2>&1
then
    echo "Error: 'perf' command not found. Please install 'linux-tools-common' or equivalent."
    exit 1
fi

# Check for python
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "Error: Python command not found at '$PYTHON_CMD'"
    echo "Please edit 'numba_benchmark_reorder.sh' and set the PYTHON_CMD variable."
    exit 1
fi
echo "Using Python command: $PYTHON_CMD"


# --- Helper Function for Perf ---
# $1: Benchmark Name
# $2: N
# $3: Reps
# $4: Output CSV File
run_perf_stat() {
    local bench_name=$1
    local N=$2
    local reps=$3
    local out_file=$4
    local avg_time="NA"
    
    # Mode is hardcoded to "multi_run_perf"
    # The Python script will handle the looping.
    local mode="multi_run_perf"

    echo "    Running PERF (N=$N, $reps reps): $bench_name"
    
    # Get the raw output from perf (which goes to stderr)
    # Note: B1 and B2 are set to 0 since loop reordering doesn't use them
    local RAW_PERF_OUTPUT=$(perf stat -x, -e $PERF_EVENTS \
                    $PYTHON_CMD numba_matmul.py \
                    --benchmark $bench_name \
                    --N $N \
                    --B1 0 \
                    --B2 0 \
                    --mode $mode \
                    --reps $reps 2>&1)
    
    # Check for perf paranoid error
    if [[ "$RAW_PERF_OUTPUT" == *"limited"* ]]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Error: 'perf' access is limited."
        echo "Your 'kernel.perf_event_paranoid' setting is too high."
        echo "Try running this script with 'sudo ./numba_benchmark_reorder.sh'"
        echo "Or run: 'sudo sysctl -w kernel.perf_event_paranoid=1'"
        echo "Aborting."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    
    # --- Check for missing kernel-specific perf tools ---
    if [[ "$RAW_PERF_OUTPUT" == *"WARNING: perf not found for kernel"* ]]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Error: 'perf' tools not found for your kernel version."
        echo "You need to install the kernel-specific perf tools."
        echo "Run one of these commands:"
        echo "  sudo apt-get install linux-tools-\$(uname -r) linux-cloud-tools-\$(uname -r)"
        echo "  sudo apt-get install linux-tools-generic linux-cloud-tools-generic"
        echo "Aborting."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    
    # --- Check for bad perf events ---
    if [[ "$RAW_PERF_OUTPUT" == *"Bad event name"* || "$RAW_PERF_OUTPUT" == *"event syntax error"* || "$RAW_PERF_OUTPUT" == *"not supported"* ]]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Error: 'perf' failed to find an event for '$bench_name' (N=$N)."
        echo "Your PERF_EVENTS list in 'numba_benchmark_reorder.sh' is likely incorrect for your CPU."
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
    local TIMING_OUTPUT=$($PYTHON_CMD numba_matmul.py \
                    --benchmark $bench_name \
                    --N $N \
                    --B1 0 \
                    --B2 0 \
                    --mode multi_run_timing \
                    --reps $reps 2>&1 )

    if [[ $TIMING_OUTPUT =~ avg_time_sec=([0-9eE\.+-]+) ]]; then
        avg_time=${BASH_REMATCH[1]}
    else
        echo "Warning: Unable to parse timing output for $bench_name (N=$N)." >&2
        echo "Raw timing output:" >&2
        echo "$TIMING_OUTPUT" >&2
        avg_time="NA"
    fi

    # Write the final, clean CSV row
    echo "$bench_name,$N,$reps,$PERF_VALUES,$avg_time" >> $out_file
}

# ===================================================================
# --- Main Benchmark Loop ---
# ===================================================================
echo -e "\n--- Running 'perf stat' for Loop Reordering Benchmarks ---"

# Write the CSV header
echo "benchmark_name,N,reps,$PERF_EVENTS,avg_time_sec" > $OUT_CSV

for N in $N_SIZES; do
    echo "--- Benchmarking Loop Reordering for N=$N ($REPS reps) ---"
    
    # --- Baseline: NumPy and Naive ---
    echo "  Benchmarking Baseline..."
    run_perf_stat "numpy_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_naive_mul" $N $REPS $OUT_CSV
    
    # --- All 6 Loop Orderings (Sequential) ---
    echo "  Benchmarking Loop Reorderings (Sequential)..."
    run_perf_stat "numba_reordered_ijk_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_reordered_ikj_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_reordered_jik_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_reordered_jki_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_reordered_kij_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_reordered_kji_mul" $N $REPS $OUT_CSV
    
    # --- All 6 Loop Orderings (Parallel) ---
    echo "  Benchmarking Loop Reorderings (Parallel)..."
    run_perf_stat "numba_parallel_naive_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_parallel_reordered_ijk_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_parallel_reordered_ikj_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_parallel_reordered_jik_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_parallel_reordered_jki_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_parallel_reordered_kij_mul" $N $REPS $OUT_CSV
    run_perf_stat "numba_parallel_reordered_kji_mul" $N $REPS $OUT_CSV
done

echo -e "\n--- Loop Reordering benchmarks complete. ---"
echo "Results are in:"
echo "  - $OUT_CSV (Detailed CSV for analysis)"
echo "Done."
