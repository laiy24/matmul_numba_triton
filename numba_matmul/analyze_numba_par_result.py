import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_FILENAME = "numba_benchmark_par_results.csv"
SAVE_PLOTS = True
PLOT_DIR = "numba_plots"

THREAD_METRIC_ORDER = [
    ("L1-miss-rate", "L1 Miss Rate"),
    ("L2-miss-rate", "L2 Miss Rate"),
    ("LLC-miss-rate", "LLC Miss Rate"),
    ("avg_time_sec", "Average Runtime (s)")
]

NUMERIC_COLS = [
    "threads",
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

    if "threads" in df.columns:
        df["threads"] = df["threads"].astype(pd.Int64Dtype())

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


def plot_thread_effect(df: pd.DataFrame) -> None:
    data = df.sort_values("threads").dropna(subset=["threads"])

    if data.empty:
        print("No data to plot. Make sure the benchmark CSV is populated.")
        return

    ensure_plot_dir(PLOT_DIR)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Thread Count Effect on numba_parallel_tiled_mul_transposed", fontsize=15, y=1.02)

    for (metric, title), ax in zip(THREAD_METRIC_ORDER, axes.flat):
        if metric not in data.columns:
            ax.set_visible(False)
            continue

        sns.lineplot(data=data, x="threads", y=metric, marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("NUMBA_NUM_THREADS")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if metric == "avg_time_sec":
            ax.set_ylabel("Seconds (lower is better)")
        else:
            ax.set_ylabel("Ratio")

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if SAVE_PLOTS:
        output_path = os.path.join(PLOT_DIR, "thread_count_effect.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show(block=False)
    plt.pause(0.1)
    plt.close(fig)


def print_best_configuration(df: pd.DataFrame) -> None:
    if "avg_time_sec" not in df.columns:
        return

    data = df.dropna(subset=["avg_time_sec"])
    if data.empty:
        return

    best_row = data.loc[data["avg_time_sec"].idxmin()]
    threads = int(best_row.get("threads", -1))
    avg_time = best_row.get("avg_time_sec", float("nan"))
    print(f"Fastest average runtime: {avg_time:.6f}s at NUMBA_NUM_THREADS={threads}")


def main() -> None:
    df = load_dataframe(CSV_FILENAME)
    if df.empty:
        return

    plot_thread_effect(df)
    print_best_configuration(df)


if __name__ == "__main__":
    main()
