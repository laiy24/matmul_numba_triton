import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_FILENAME = "numba_benchmark_reorder_results.csv"
SAVE_PLOTS = True
PLOT_DIR = "numba_plots"

NUMERIC_COLS = [
    "N",
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

    # Filter out numpy benchmarks - only keep reordered benchmarks
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


def extract_loop_order(benchmark_name: str) -> str:
    """Extract loop order from benchmark name. Returns the loop ordering (ijk, ikj, etc.)."""
    if "reordered_ijk" in benchmark_name:
        return "ijk"
    elif "reordered_ikj" in benchmark_name:
        return "ikj"
    elif "reordered_jik" in benchmark_name:
        return "jik"
    elif "reordered_jki" in benchmark_name:
        return "jki"
    elif "reordered_kij" in benchmark_name:
        return "kij"
    elif "reordered_kji" in benchmark_name:
        return "kji"
    elif "naive" in benchmark_name:
        return "naive"
    else:
        return "other"


def categorize_benchmark(benchmark_name: str) -> str:
    """Categorize benchmark as sequential or parallel."""
    if "parallel" in benchmark_name:
        return "parallel"
    else:
        return "sequential"


def plot_reorder_comparison_by_size(df: pd.DataFrame) -> None:
    """Plot the effect of loop reordering for each matrix size."""
    ensure_plot_dir(PLOT_DIR)
    
    # Add derived columns
    df["loop_order"] = df["benchmark_name"].apply(extract_loop_order)
    df["category"] = df["benchmark_name"].apply(categorize_benchmark)
    
    # Work with numba implementations only (numpy already filtered in load_dataframe)
    numba_data = df[df["category"].isin(["sequential", "parallel"])].copy()
    
    if numba_data.empty:
        print("No data to plot.")
        return
    
    # Get unique matrix sizes
    sizes = sorted(numba_data["N"].unique())
    
    # Define loop order for consistent plotting
    loop_orders = ["naive", "ijk", "ikj", "jik", "jki", "kij", "kji"]
    
    for size in sizes:
        size_data = numba_data[numba_data["N"] == size].copy()
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Loop Reordering Effect (N={size})", fontsize=16, y=0.995)
        
        # Separate sequential and parallel
        seq_data = size_data[size_data["category"] == "sequential"].copy()
        par_data = size_data[size_data["category"] == "parallel"].copy()
        
        # Sort by loop order
        seq_data["loop_order_cat"] = pd.Categorical(seq_data["loop_order"], categories=loop_orders, ordered=True)
        par_data["loop_order_cat"] = pd.Categorical(par_data["loop_order"], categories=loop_orders, ordered=True)
        seq_data = seq_data.sort_values("loop_order_cat")
        par_data = par_data.sort_values("loop_order_cat")
        
        # Plot 1: Runtime comparison
        ax = axes[0, 0]
        x_pos_seq = np.arange(len(seq_data))
        x_pos_par = np.arange(len(par_data))
        
        if not seq_data.empty:
            ax.bar(x_pos_seq - 0.2, seq_data["avg_time_sec"], 0.4, 
                   label="Sequential", alpha=0.8, color="steelblue")
        if not par_data.empty:
            ax.bar(x_pos_par + 0.2, par_data["avg_time_sec"], 0.4, 
                   label="Parallel", alpha=0.8, color="coral")
        
        ax.set_xlabel("Loop Order")
        ax.set_ylabel("Average Runtime (s)")
        ax.set_title("Runtime vs Loop Order")
        if not seq_data.empty:
            ax.set_xticks(x_pos_seq)
            ax.set_xticklabels(seq_data["loop_order"], rotation=45)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6, axis='y')
        
        # Plot 2: IPC comparison
        ax = axes[0, 1]
        if not seq_data.empty:
            ax.bar(x_pos_seq - 0.2, seq_data["IPC"], 0.4, 
                   label="Sequential", alpha=0.8, color="steelblue")
        if not par_data.empty:
            ax.bar(x_pos_par + 0.2, par_data["IPC"], 0.4, 
                   label="Parallel", alpha=0.8, color="coral")
        
        ax.set_xlabel("Loop Order")
        ax.set_ylabel("Instructions Per Cycle")
        ax.set_title("IPC vs Loop Order")
        if not seq_data.empty:
            ax.set_xticks(x_pos_seq)
            ax.set_xticklabels(seq_data["loop_order"], rotation=45)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6, axis='y')
        
        # Plot 3: L1 miss rate
        ax = axes[1, 0]
        if not seq_data.empty:
            ax.bar(x_pos_seq - 0.2, seq_data["L1-miss-rate"], 0.4, 
                   label="Sequential", alpha=0.8, color="steelblue")
        if not par_data.empty:
            ax.bar(x_pos_par + 0.2, par_data["L1-miss-rate"], 0.4, 
                   label="Parallel", alpha=0.8, color="coral")
        
        ax.set_xlabel("Loop Order")
        ax.set_ylabel("L1 Miss Rate")
        ax.set_title("L1 Cache Miss Rate vs Loop Order")
        if not seq_data.empty:
            ax.set_xticks(x_pos_seq)
            ax.set_xticklabels(seq_data["loop_order"], rotation=45)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6, axis='y')
        
        # Plot 4: L2 miss rate
        ax = axes[1, 1]
        if not seq_data.empty:
            ax.bar(x_pos_seq - 0.2, seq_data["L2-miss-rate"], 0.4, 
                   label="Sequential", alpha=0.8, color="steelblue")
        if not par_data.empty:
            ax.bar(x_pos_par + 0.2, par_data["L2-miss-rate"], 0.4, 
                   label="Parallel", alpha=0.8, color="coral")
        
        ax.set_xlabel("Loop Order")
        ax.set_ylabel("L2 Miss Rate")
        ax.set_title("L2 Cache Miss Rate vs Loop Order")
        if not seq_data.empty:
            ax.set_xticks(x_pos_seq)
            ax.set_xticklabels(seq_data["loop_order"], rotation=45)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6, axis='y')
        
        plt.tight_layout()
        
        if SAVE_PLOTS:
            output_path = os.path.join(PLOT_DIR, f"reorder_effect_N{size}.png")
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)


def plot_cache_behavior_heatmap(df: pd.DataFrame) -> None:
    """Create heatmaps showing cache behavior for different loop orders."""
    ensure_plot_dir(PLOT_DIR)
    
    df["loop_order"] = df["benchmark_name"].apply(extract_loop_order)
    df["category"] = df["benchmark_name"].apply(categorize_benchmark)
    
    sizes = sorted(df["N"].unique())
    loop_orders = ["ijk", "ikj", "jik", "jki", "kij", "kji"]
    
    for size in sizes:
        size_data = df[df["N"] == size].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Cache Miss Rates by Loop Order (N={size})", fontsize=16, y=1.00)
        
        for cat_idx, category in enumerate(["sequential", "parallel"]):
            cat_data = size_data[size_data["category"] == category].copy()
            cat_data = cat_data[cat_data["loop_order"].isin(loop_orders)]
            
            if cat_data.empty:
                axes[cat_idx].set_visible(False)
                continue
            
            # Sort by loop order
            cat_data["loop_order_cat"] = pd.Categorical(cat_data["loop_order"], categories=loop_orders, ordered=True)
            cat_data = cat_data.sort_values("loop_order_cat")
            
            # Create heatmap data
            heatmap_data = cat_data[["L1-miss-rate", "L2-miss-rate", "LLC-miss-rate"]].T
            heatmap_data.columns = cat_data["loop_order"]
            
            sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlOrRd", 
                       ax=axes[cat_idx], cbar_kws={"label": "Miss Rate"})
            axes[cat_idx].set_title(f"{category.capitalize()}")
            axes[cat_idx].set_xlabel("Loop Order")
            axes[cat_idx].set_ylabel("Cache Level")
        
        plt.tight_layout()
        
        if SAVE_PLOTS:
            output_path = os.path.join(PLOT_DIR, f"reorder_cache_heatmap_N{size}.png")
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a summary table of best configurations."""
    df["loop_order"] = df["benchmark_name"].apply(extract_loop_order)
    df["category"] = df["benchmark_name"].apply(categorize_benchmark)
    
    print("\n" + "="*80)
    print("SUMMARY: Best Loop Reordering Configuration by Matrix Size")
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
            print(f"    Loop Order: {best_seq['loop_order']}")
            print(f"    Runtime: {best_seq['avg_time_sec']:.6f}s")
            print(f"    IPC: {best_seq['IPC']:.3f}")
            print(f"    L1 Miss Rate: {best_seq['L1-miss-rate']:.4f}")
            print(f"    L2 Miss Rate: {best_seq['L2-miss-rate']:.4f}")
            
            # Also show naive for comparison
            naive_seq = seq_data[seq_data["loop_order"] == "naive"]
            if not naive_seq.empty:
                naive = naive_seq.iloc[0]
                speedup = naive["avg_time_sec"] / best_seq["avg_time_sec"]
                print(f"    Speedup vs Naive: {speedup:.2f}x")
        
        # Parallel best
        par_data = size_data[size_data["category"] == "parallel"]
        if not par_data.empty:
            best_par = par_data.loc[par_data["avg_time_sec"].idxmin()]
            print(f"  Parallel Best: {best_par['benchmark_name']}")
            print(f"    Loop Order: {best_par['loop_order']}")
            print(f"    Runtime: {best_par['avg_time_sec']:.6f}s")
            print(f"    IPC: {best_par['IPC']:.3f}")
            print(f"    L1 Miss Rate: {best_par['L1-miss-rate']:.4f}")
            print(f"    L2 Miss Rate: {best_par['L2-miss-rate']:.4f}")
            
            # Also show naive for comparison
            naive_par = par_data[par_data["loop_order"] == "naive"]
            if not naive_par.empty:
                naive = naive_par.iloc[0]
                speedup = naive["avg_time_sec"] / best_par["avg_time_sec"]
                print(f"    Speedup vs Naive: {speedup:.2f}x")
    
    print("="*80 + "\n")


def print_loop_order_ranking(df: pd.DataFrame) -> None:
    """Print ranking of loop orders by performance."""
    df["loop_order"] = df["benchmark_name"].apply(extract_loop_order)
    df["category"] = df["benchmark_name"].apply(categorize_benchmark)
    
    print("\n" + "="*80)
    print("LOOP ORDER RANKING (Best to Worst by Runtime)")
    print("="*80)
    
    sizes = sorted(df["N"].unique())
    
    for size in sizes:
        size_data = df[df["N"] == size].copy()
        
        print(f"\nMatrix Size N={size}:")
        print("-" * 80)
        
        for category in ["sequential", "parallel"]:
            cat_data = size_data[size_data["category"] == category].copy()
            if cat_data.empty:
                continue
            
            cat_data = cat_data.sort_values("avg_time_sec")
            
            print(f"\n  {category.capitalize()}:")
            for i, (_, row) in enumerate(cat_data.iterrows(), 1):
                print(f"    {i}. {row['loop_order']:6s} - {row['avg_time_sec']:10.6f}s "
                      f"(IPC: {row['IPC']:.3f}, L1-miss: {row['L1-miss-rate']:.4f})")
    
    print("="*80 + "\n")


def main() -> None:
    df = load_dataframe(CSV_FILENAME)
    if df.empty:
        return

    plot_reorder_comparison_by_size(df)
    plot_cache_behavior_heatmap(df)
    print_summary_table(df)
    print_loop_order_ranking(df)


if __name__ == "__main__":
    main()
