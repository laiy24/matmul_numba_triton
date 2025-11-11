"""Analyze the impact of NUMBA_SLP_VECTORIZE on tiled matmul benchmarks."""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CSV_FILENAME = "numba_benchmark_vec_results.csv"
PLOT_DIR = "numba_plots_vectorize"
RUNTIME_PLOT = "vectorization_runtime.png"
SPEEDUP_PLOT = "vectorization_speedup.png"


def _prepare_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NA", "nan"])

    numeric_cols = [
        "vectorize_enabled",
        "N",
        "B1",
        "B2",
        "reps",
        "cycles",
        "instructions",
        "L1-dcache-loads",
        "L1-dcache-load-misses",
        "L2-loads",
        "L2-load-misses",
        "LLC-loads",
        "LLC-load-misses",
        "avg_time_sec",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["vectorize_enabled", "B1", "avg_time_sec"])

    df["vectorization"] = df["vectorize_enabled"].map(
        {0: "Vectorize Off", 1: "Vectorize On"}
    )

    return df


def _ensure_plot_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _plot_runtime(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="B1",
        y="avg_time_sec",
        hue="vectorization",
        marker="o",
        ax=ax,
    )

    ax.set_title("Runtime vs. Block Size (Numba Tiled MatMul)")
    ax.set_xlabel("Block Size B1")
    ax.set_ylabel("Average Time (s)")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.legend(title="NUMBA_SLP_VECTORIZE")

    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, RUNTIME_PLOT)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved runtime plot: {output_path}")


def _plot_speedup(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(
        index="B1",
        columns="vectorize_enabled",
        values="avg_time_sec",
        aggfunc="min",
    )

    if 0 not in pivot.columns or 1 not in pivot.columns:
        print("Warning: Missing vectorization states for speedup plot; skipping.")
        return

    pivot["speedup"] = pivot[0] / pivot[1]
    pivot = pivot.reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=pivot, x="B1", y="speedup", ax=ax, color="#4C72B0")
    ax.set_title("Speedup from NUMBA_SLP_VECTORIZE (Higher is Better)")
    ax.set_xlabel("Block Size B1")
    ax.set_ylabel("Scalar Runtime / Vectorized Runtime")

    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, SPEEDUP_PLOT)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved speedup plot: {output_path}")

    best_row = pivot.loc[pivot["speedup"].idxmax()]
    print(
        "Best observed speedup: B1={B1}, speedup={speedup:.2f}x".format(
            B1=int(best_row["B1"]), speedup=best_row["speedup"]
        )
    )


def main() -> None:
    try:
        df = _prepare_dataframe(CSV_FILENAME)
    except FileNotFoundError:
        print(
            f"Error: '{CSV_FILENAME}' not found. Run numba_benchmark_vec.sh before analysing.",
            file=sys.stderr,
        )
        sys.exit(1)

    if df.empty:
        print("No valid benchmark rows found in the CSV.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from '{CSV_FILENAME}'.")

    _ensure_plot_dir(PLOT_DIR)
    _plot_runtime(df)
    _plot_speedup(df)

    summary = (
        df.groupby(["vectorization", "B1"], as_index=False)["avg_time_sec"]
        .min()
        .sort_values("avg_time_sec")
    )

    print("\nFastest configurations (lower time first):")
    print(summary.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
