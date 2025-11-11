#!/bin/bash

echo "--- Starting NUMBA SLP Vectorization Benchmark ---"

# --- Configuration ---
# Paste the full path to the python executable from your conda or venv environment
PYTHON_CMD="/home/laiy24/miniconda3/envs/numba/bin/python"

# Number of times to repeat each configuration inside the Python runner
REPS=20

# Benchmark focus
N_SIZE=2048
B_SIZES="4 8 16 32 64 128"
BENCHMARK="numba_tiled_mul"

# Perf events to monitor (adjust if unsupported on your CPU)
PERF_EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L2-loads,L2-load-misses,LLC-loads,LLC-load-misses"

# Output file
OUT_CSV="numba_benchmark_vec_results.csv"

# --- Setup ---
rm -f "$OUT_CSV"

if ! command -v perf &> /dev/null; then
    echo "Error: 'perf' command not found. Please install 'linux-tools-common' or equivalent."
    exit 1
fi

if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "Error: Python command not found at '$PYTHON_CMD'"
    echo "Please edit 'numba_benchmark_vec.sh' and set the PYTHON_CMD variable."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

echo "benchmark_name,vectorize_enabled,N,B1,B2,reps,$PERF_EVENTS,avg_time_sec" > "$OUT_CSV"

run_vectorization_stat() {
    local vector_flag=$1
    local N=$2
    local B1=$3
    local B2=$4
    local reps=$5
    local out_file=$6

    local avg_time="NA"
    local PERF_VALUES="NA,NA,NA,NA,NA,NA,NA,NA"

    echo "    Running NUMBA_SLP_VECTORIZE=$vector_flag (N=$N, B1=$B1, reps=$reps)"

    local RAW_PERF_OUTPUT=$(NUMBA_SLP_VECTORIZE=$vector_flag perf stat -x, -e $PERF_EVENTS \
        "$PYTHON_CMD" numba_matmul.py \
        --benchmark $BENCHMARK \
        --N $N \
        --B1 $B1 \
        --B2 $B2 \
        --mode multi_run_perf \
        --reps $reps 2>&1)

    if [[ "$RAW_PERF_OUTPUT" == *"limited"* ]]; then
        echo "Error: 'perf' access is limited. Adjust 'kernel.perf_event_paranoid' or run with sudo."
        exit 1
    fi

    if [[ "$RAW_PERF_OUTPUT" == *"Bad event name"* || "$RAW_PERF_OUTPUT" == *"event syntax error"* || "$RAW_PERF_OUTPUT" == *"not supported"* ]]; then
        echo "Warning: One or more perf events failed for NUMBA_SLP_VECTORIZE=$vector_flag (B1=$B1). Writing NA values."
        echo "Raw perf error: $RAW_PERF_OUTPUT"
    else
        PERF_VALUES=$(echo "$RAW_PERF_OUTPUT" | grep -v "^[ \t]*$" | awk -F, '{ print $1 }' | paste -sd,)
    fi

    local TIMING_OUTPUT=$(NUMBA_SLP_VECTORIZE=$vector_flag "$PYTHON_CMD" numba_matmul.py \
        --benchmark $BENCHMARK \
        --N $N \
        --B1 $B1 \
        --B2 $B2 \
        --mode multi_run_timing \
        --reps $reps 2>&1)

    if [[ $TIMING_OUTPUT =~ avg_time_sec=([0-9eE.+-]+) ]]; then
        avg_time=${BASH_REMATCH[1]}
    else
        echo "Warning: Unable to parse timing output for NUMBA_SLP_VECTORIZE=$vector_flag (B1=$B1)." >&2
        echo "Raw timing output:" >&2
        echo "$TIMING_OUTPUT" >&2
        avg_time="NA"
    fi

    echo "$BENCHMARK,$vector_flag,$N,$B1,$B2,$reps,$PERF_VALUES,$avg_time" >> "$out_file"
}

echo "\n--- Benchmarking N=$N_SIZE with varying B1 ---"

for vector_flag in 0 1; do
    echo "\n=== NUMBA_SLP_VECTORIZE=$vector_flag ==="
    for B1 in $B_SIZES; do
        run_vectorization_stat $vector_flag $N_SIZE $B1 0 $REPS "$OUT_CSV"
    done
done

echo -e "\n--- Vectorization benchmark complete. ---"
echo "Results are in: $OUT_CSV"
