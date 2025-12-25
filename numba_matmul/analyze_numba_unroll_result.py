import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_FILENAME = "numba_benchmark_loop_unroll_results.csv"
SAVE_PLOTS = True
PLOT_DIR = "numba_plots"

NUMERIC_COLS = [
    "N",
    "B1",
    "reps",
    "cycles",
    "instructions",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "L2-loads",
    "L2-load-misses",
    "LLC-loads",
    "LLC-load-misses",
    "avg_time_sec"
]


def load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Error: CSV file '{path}' not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cycles" in df.columns and "instructions" in df.columns:
        df["IPC"] = (
            df["instructions"].div(df["cycles"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    else:
        df["IPC"] = np.nan

    df["L1-miss-rate"] = safe_ratio(df, "L1-dcache-load-misses", "L1-dcache-loads")
    df["L2-miss-rate"] = safe_ratio(df, "L2-load-misses", "L2-loads")
    df["LLC-miss-rate"] = safe_ratio(df, "LLC-load-misses", "LLC-loads")

    # Filter out numpy benchmarks - only keep unroll and naive benchmarks
    if "benchmark_name" in df.columns:
        df = df[~df["benchmark_name"].str.contains("numpy", case=False, na=False)].copy()

    return df


def safe_ratio(df: pd.DataFrame, num: str, den: str) -> pd.Series:
    if num not in df.columns or den not in df.columns:
        return pd.Series(np.zeros(len(df)))
    return (
        df[num]
        .div(df[den])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def ensure_plot_dir(path: str) -> None:
    if SAVE_PLOTS and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_unroll_factor(benchmark_name: str) -> int:
    """Extract unroll factor from benchmark name. Returns 1 for non-unrolled."""
    if "unrolled2" in benchmark_name:
        return 2
    elif "unrolled4" in benchmark_name:
        return 4
    elif "unrolled8" in benchmark_name:
        return 8
    elif "unrolled16" in benchmark_name:
        return 16
    elif "reordered" in benchmark_name:
        return 1  # Reordered but not unrolled
    elif "naive" in benchmark_name:
        return 0  # Baseline
    else:
        return -1  # Other (e.g., numpy)


def categorize_benchmark(benchmark_name: str) -> str:
    """Categorize benchmark as sequential or parallel."""
    if "parallel" in benchmark_name:
        return "parallel"
    else:
        return "sequential"


def plot_unroll_comparison_by_size(df: pd.DataFrame) -> None:
    """Plot the effect of loop unrolling for each matrix size."""
    ensure_plot_dir(PLOT_DIR)
    
    # Add derived columns
    df["unroll_factor"] = df["benchmark_name"].apply(extract_unroll_factor)
    df["category"] = df["benchmark_name"].apply(categorize_benchmark)
    
    # Work with numba implementations only (numpy already filtered in load_dataframe)
    numba_data = df[df["category"].isin(["sequential", "parallel"])].copy()
    
    if numba_data.empty:
        print("No data to plot.")
        return
    
    # Get unique matrix sizes
    sizes = sorted(numba_data["N"].unique())
    
    for size in sizes:
        size_data = numba_data[numba_data["N"] == size].copy()
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Loop Unrolling Effect (N={size})", fontsize=16, y=1.00)
        
        # Separate sequential and parallel
        seq_data = size_data[size_data["category"] == "sequential"].sort_values("unroll_factor")
        par_data = size_data[size_data["category"] == "parallel"].sort_values("unroll_factor")
        
        # Plot 1: Runtime comparison
        ax = axes[0, 0]
        if not seq_data.empty:
            ax.plot(seq_data["unroll_factor"], seq_data["avg_time_sec"], 
                   marker="o", label="Sequential", linewidth=2)
        if not par_data.empty:
            ax.plot(par_data["unroll_factor"], par_data["avg_time_sec"], 
                   marker="s", label="Parallel", linewidth=2)
        ax.set_xlabel("Unroll Factor (0=naive, 1=reordered)")
        ax.set_ylabel("Average Runtime (s)")
        ax.set_title("Runtime vs Unroll Factor")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Plot 2: IPC comparison
        ax = axes[0, 1]
        if not seq_data.empty:
            ax.plot(seq_data["unroll_factor"], seq_data["IPC"], 
                   marker="o", label="Sequential", linewidth=2)
        if not par_data.empty:
            ax.plot(par_data["unroll_factor"], par_data["IPC"], 
                   marker="s", label="Parallel", linewidth=2)
        ax.set_xlabel("Unroll Factor (0=naive, 1=reordered)")
        ax.set_ylabel("Instructions Per Cycle")
        ax.set_title("IPC vs Unroll Factor")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Plot 3: L1 miss rate
        ax = axes[1, 0]
        if not seq_data.empty:
            ax.plot(seq_data["unroll_factor"], seq_data["L1-miss-rate"], 
                   marker="o", label="Sequential", linewidth=2)
        if not par_data.empty:
            ax.plot(par_data["unroll_factor"], par_data["L1-miss-rate"], 
                   marker="s", label="Parallel", linewidth=2)
        ax.set_xlabel("Unroll Factor (0=naive, 1=reordered)")
        ax.set_ylabel("L1 Miss Rate")
        ax.set_title("L1 Cache Miss Rate vs Unroll Factor")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Plot 4: L2 miss rate
        ax = axes[1, 1]
        if not seq_data.empty:
            ax.plot(seq_data["unroll_factor"], seq_data["L2-miss-rate"], 
                   marker="o", label="Sequential", linewidth=2)
        if not par_data.empty:
            ax.plot(par_data["unroll_factor"], par_data["L2-miss-rate"], 
                   marker="s", label="Parallel", linewidth=2)
        ax.set_xlabel("Unroll Factor (0=naive, 1=reordered)")
        ax.set_ylabel("L2 Miss Rate")
        ax.set_title("L2 Cache Miss Rate vs Unroll Factor")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if SAVE_PLOTS:
            output_path = os.path.join(PLOT_DIR, f"unroll_effect_N{size}.png")
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)


def plot_speedup_comparison(df: pd.DataFrame) -> None:
    """Plot speedup relative to naive implementation."""
    ensure_plot_dir(PLOT_DIR)
    
    df["unroll_factor"] = df["benchmark_name"].apply(extract_unroll_factor)
    df["category"] = df["benchmark_name"].apply(categorize_benchmark)
    
    sizes = sorted(df["N"].unique())
    
    fig, axes = plt.subplots(1, len(sizes), figsize=(7 * len(sizes), 6))
    if len(sizes) == 1:
        axes = [axes]
    
    fig.suptitle("Speedup vs Naive Implementation", fontsize=16, y=1.00)
    
    for idx, size in enumerate(sizes):
        size_data = df[df["N"] == size].copy()
        
        # Get baseline (naive) runtime for sequential and parallel
        seq_naive = size_data[size_data["benchmark_name"] == "numba_naive_mul"]
        par_naive = size_data[size_data["benchmark_name"] == "numba_parallel_naive_mul"]
        
        if seq_naive.empty and par_naive.empty:
            axes[idx].set_visible(False)
            continue
        
        seq_baseline_time = seq_naive["avg_time_sec"].values[0] if not seq_naive.empty else None
        par_baseline_time = par_naive["avg_time_sec"].values[0] if not par_naive.empty else None
        
        # Calculate speedup for sequential variants
        seq_data = size_data[size_data["category"] == "sequential"].copy()
        if not seq_data.empty and seq_baseline_time:
            seq_data["speedup"] = seq_baseline_time / seq_data["avg_time_sec"]
            seq_data = seq_data.sort_values("unroll_factor")
            axes[idx].plot(seq_data["unroll_factor"], seq_data["speedup"], 
                          marker="o", label="Sequential", linewidth=2, markersize=8)
        
        # Calculate speedup for parallel variants
        par_data = size_data[size_data["category"] == "parallel"].copy()
        if not par_data.empty and par_baseline_time:
            par_data["speedup"] = par_baseline_time / par_data["avg_time_sec"]
            par_data = par_data.sort_values("unroll_factor")
            axes[idx].plot(par_data["unroll_factor"], par_data["speedup"], 
                          marker="s", label="Parallel", linewidth=2, markersize=8)
        
        axes[idx].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label="Baseline")
        axes[idx].set_xlabel("Unroll Factor (0=naive, 1=reordered)")
        axes[idx].set_ylabel("Speedup")
        axes[idx].set_title(f"N={size}")
        axes[idx].legend()
        axes[idx].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if SAVE_PLOTS:
        output_path = os.path.join(PLOT_DIR, "unroll_speedup_comparison.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    
    plt.show(block=False)
    plt.pause(0.1)
    plt.close(fig)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a summary table of best configurations."""
    df["unroll_factor"] = df["benchmark_name"].apply(extract_unroll_factor)
    df["category"] = df["benchmark_name"].apply(categorize_benchmark)
    
    print("\n" + "="*80)
    print("SUMMARY: Best Loop Unrolling Configuration by Matrix Size")
    print("="*80)
    
    sizes = sorted(df["N"].unique())
    
    for size in sizes:
        size_data = df[df["N"] == size].copy()
        
        print(f"\nMatrix Size N={size}:")
        print("-" * 80)
        
        # Sequential best
        seq_data = size_data[size_data["category"] == "sequential"]
        if not seq_data.empty:
            best_seq = seq_data.loc[seq_data["avg_time_sec"].idxmin()]
            print(f"  Sequential Best: {best_seq['benchmark_name']}")
            print(f"    Unroll Factor: {int(best_seq['unroll_factor'])}")
            print(f"    Runtime: {best_seq['avg_time_sec']:.6f}s")
            print(f"    IPC: {best_seq['IPC']:.3f}")
            print(f"    L1 Miss Rate: {best_seq['L1-miss-rate']:.4f}")
        
        # Parallel best
        par_data = size_data[size_data["category"] == "parallel"]
        if not par_data.empty:
            best_par = par_data.loc[par_data["avg_time_sec"].idxmin()]
            print(f"  Parallel Best: {best_par['benchmark_name']}")
            print(f"    Unroll Factor: {int(best_par['unroll_factor'])}")
            print(f"    Runtime: {best_par['avg_time_sec']:.6f}s")
            print(f"    IPC: {best_par['IPC']:.3f}")
            print(f"    L1 Miss Rate: {best_par['L1-miss-rate']:.4f}")
    
    print("="*80 + "\n")


def main() -> None:
    df = load_dataframe(CSV_FILENAME)
    if df.empty:
        return

    plot_unroll_comparison_by_size(df)
    plot_speedup_comparison(df)
    print_summary_table(df)


if __name__ == "__main__":
    main()
