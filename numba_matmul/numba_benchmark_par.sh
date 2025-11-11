#!/bin/bash

echo "--- Starting Thread Count Benchmark ---"

# --- Configuration ---
# Update this path to point at the interpreter inside your environment.
PYTHON_CMD="/home/isabelle/miniconda3/envs/numba/bin/python"

# Repeat count inside each Python invocation.
REPS=20

# Thread counts to compare.
THREAD_COUNTS="2 4 6 8 10 14 16 24 32"

# Fixed workload parameters.
N_SIZE=2048
B1_SIZE=32

# Perf events to collect. Adjust if unsupported on your CPU.
PERF_EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L2-loads,L2-load-misses,LLC-loads,LLC-load-misses"

# Output CSV.
OUT_CSV="numba_benchmark_par_results.csv"

# --- Setup ---
rm -f "$OUT_CSV"

echo "Checking dependencies..."
if ! command -v perf >/dev/null 2>&1; then
    echo "Error: 'perf' command not found. Install linux-tools-common or your distro equivalent."
    exit 1
fi

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo "Error: Python command not found at '$PYTHON_CMD'"
    echo "Edit numba_benchmark_par.sh and set PYTHON_CMD to your interpreter."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

echo "benchmark_name,threads,N,B1,reps,$PERF_EVENTS,avg_time_sec" > "$OUT_CSV"

declare -a PERF_EVENT_ARRAY
IFS=',' read -ra PERF_EVENT_ARRAY <<< "$PERF_EVENTS"

run_perf_stat() {
    local threads=$1
    local bench_name=$2
    local N=$3
    local B1=$4
    local reps=$5
    local out_file=$6
    local mode="multi_run_perf"
    local avg_time="NA"

    echo "  Threads=$threads : running perf"

    local raw_perf_output
    raw_perf_output=$(NUMBA_NUM_THREADS=$threads perf stat -x, -e "$PERF_EVENTS" \
        "$PYTHON_CMD" numba_matmul.py \
        --benchmark "$bench_name" \
        --N "$N" \
        --B1 "$B1" \
        --mode "$mode" \
        --reps "$reps" 2>&1)

    if [[ "$raw_perf_output" == *"limited"* ]]; then
        echo "Perf access denied. Run with sudo or adjust kernel.perf_event_paranoid."
        exit 1
    fi

    local perf_values
    if [[ "$raw_perf_output" == *"Bad event name"* || "$raw_perf_output" == *"event syntax error"* || "$raw_perf_output" == *"not supported"* ]]; then
        echo "Perf event mismatch detected. Writing NA values."
        local perf_na_values=()
        for _ in "${PERF_EVENT_ARRAY[@]}"; do
            perf_na_values+=("NA")
        done
        local old_ifs="$IFS"
        IFS=','
        perf_values="${perf_na_values[*]}"
        IFS="$old_ifs"
    else
        perf_values=$(echo "$raw_perf_output" | grep -v "^[ \t]*$" | awk -F, '{ print $1 }' | paste -sd,)
    fi

    echo "  Threads=$threads : timing runs"

    local timing_output
    timing_output=$(NUMBA_NUM_THREADS=$threads "$PYTHON_CMD" numba_matmul.py \
        --benchmark "$bench_name" \
        --N "$N" \
        --B1 "$B1" \
        --mode multi_run_timing \
        --reps "$reps" 2>&1)

    if [[ $timing_output =~ avg_time_sec=([0-9eE.+-]+) ]]; then
        avg_time=${BASH_REMATCH[1]}
    else
        echo "Warning: unable to parse timing output for threads=$threads" >&2
        echo "$timing_output" >&2
        avg_time="NA"
    fi

    echo "$bench_name,$threads,$N,$B1,$reps,$perf_values,$avg_time" >> "$out_file"
}

echo "--- Running numba_parallel_tiled_mul_transposed (N=$N_SIZE, B1=$B1_SIZE) ---"
for threads in $THREAD_COUNTS; do
    echo "--- NUMBA_NUM_THREADS=$threads ---"
    run_perf_stat "$threads" "numba_parallel_tiled_mul_transposed" "$N_SIZE" "$B1_SIZE" "$REPS" "$OUT_CSV"
done

echo "--- Thread benchmark complete ---"
echo "Results written to $OUT_CSV"
