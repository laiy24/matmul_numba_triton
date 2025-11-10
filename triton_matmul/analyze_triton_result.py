#!/usr/bin/env python3
"""Generate per-size runtime bar charts from Triton benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import matplotlib


def _configure_matplotlib() -> None:
	"""Select a non-interactive backend for headless environments."""

	matplotlib.use("Agg")


_configure_matplotlib()

import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
import pandas as pd  # noqa: E402


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Create bar plots of average runtimes for every benchmark configuration "
			"grouped by the tested matrix size N."
		)
	)
	parser.add_argument(
		"--csv",
		type=Path,
		default=Path("triton_benchmark_results.csv"),
		help="Path to the CSV file produced by triton_benchmark.sh",
	)
	parser.add_argument(
		"--outdir",
		type=Path,
		default=Path("triton_plot"),
		help="Directory where the bar chart images will be saved",
	)
	return parser.parse_args()


def _load_results(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path, na_values=["NA", "nan", ""], keep_default_na=True)
	required_columns = {
		"benchmark_name",
		"N",
		"block_size_m",
		"block_size_n",
		"block_size_k",
		"avg_time_sec",
	}
	missing = required_columns - set(df.columns)
	if missing:
		missing_cols = ", ".join(sorted(missing))
		raise ValueError(f"CSV file is missing required columns: {missing_cols}")

	df = df.dropna(subset=["avg_time_sec"])  # Ignore rows without a timing result

	df["avg_time_sec"] = df["avg_time_sec"].astype(float)
	df["N"] = df["N"].astype(int)

	return df


def _format_config_label_row(row: pd.Series) -> str:
	name = str(row["benchmark_name"])
	m = row["block_size_m"]
	n = row["block_size_n"]
	k = row["block_size_k"]
	if pd.isna(m) and pd.isna(n) and pd.isna(k):
		return name

	def _fmt(value: Any) -> str:
		if pd.isna(value):
			return "NA"
		value_float = float(value)
		if value_float.is_integer():
			return str(int(value_float))
		return str(value)

	return f"{name} (m={_fmt(m)}, n={_fmt(n)}, k={_fmt(k)})"


def _aggregate_configs(df: pd.DataFrame) -> pd.DataFrame:
	grouped = (
		df.groupby(
			["N", "benchmark_name", "block_size_m", "block_size_n", "block_size_k"],
			dropna=False,
			sort=False,
		)["avg_time_sec"].mean()
	)
	aggregated = grouped.reset_index()
	aggregated["config_label"] = aggregated.apply(_format_config_label_row, axis=1)
	return aggregated


def _plot_group(
	labels: Iterable[str],
	times: Iterable[float],
	title: str,
	output_path: Path,
) -> None:
	labels_list = list(labels)
	times_list = list(times)
	if not labels_list:
		return

	width = max(6.0, 0.55 * len(labels_list))
	fig, ax = plt.subplots(figsize=(width, 4.5))

	ax.bar(range(len(times_list)), times_list, color="#1f77b4")
	ax.set_xticks(range(len(labels_list)))
	ax.set_xticklabels(labels_list, rotation=30, ha="right")
	ax.set_ylabel("Average runtime (seconds)")
	ax.set_title(title)
	ax.grid(axis="y", linestyle="--", alpha=0.2)

	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def _plot_per_size(df: pd.DataFrame, outdir: Path) -> int:
	outdir.mkdir(parents=True, exist_ok=True)

	plot_count = 0
	for n_value, group in df.groupby("N", sort=True):
		if group.empty:
			continue

		basics = group[group["benchmark_name"] == "triton_matmul_basic"]
		others = group[group["benchmark_name"] != "triton_matmul_basic"]

		basics_sorted = basics.sort_values("avg_time_sec", ascending=True)
		if not basics_sorted.empty:
			_plot_group(
				basics_sorted["config_label"],
				basics_sorted["avg_time_sec"],
				f"triton_matmul_basic block sweep (N={n_value})",
				outdir / f"runtime_basic_configs_N{n_value}.png",
			)
			plot_count += 1

		best_basic = basics_sorted.head(1)
		mix = pd.concat([others, best_basic], ignore_index=True)
		mix_sorted = mix.sort_values("avg_time_sec", ascending=True)
		if not mix_sorted.empty:
			_plot_group(
				mix_sorted["config_label"],
				mix_sorted["avg_time_sec"],
				f"All configs (best basic included) N={n_value}",
				outdir / f"runtime_mixed_configs_N{n_value}.png",
			)
			plot_count += 1

	return plot_count


def main() -> None:
	args = _parse_args()
	results = _load_results(args.csv)
	aggregated = _aggregate_configs(results)
	plot_total = _plot_per_size(aggregated, args.outdir)
	print(f"Wrote {plot_total} bar chart(s) to {args.outdir.resolve()}")


if __name__ == "__main__":
	main()

