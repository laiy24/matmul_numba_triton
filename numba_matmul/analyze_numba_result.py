import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from matplotlib.lines import Line2D

# --- Configuration ---
CSV_FILENAME = 'numba_benchmark_results.csv'  # <--- CHANGE THIS to your CSV file name
SAVE_PLOTS = True
PLOT_DIR = 'numba_plots'
DEFAULT_N_TO_PLOT = None 

# --- Helper Functions ---

def process_dataframe(df):
    """Engineers new features from the raw benchmark data."""
    df = df.copy()
    
    # Handle potential "NA" values from perf errors
    if 'avg_time_sec' not in df.columns:
        raise ValueError("CSV is missing 'avg_time_sec'. Rerun benchmarks with timing enabled.")

    numeric_cols = [
        'reps',
        'cycles', 'instructions',
        'L1-dcache-loads', 'L1-dcache-load-misses',
        'L2-loads', 'L2-load-misses',
        'LLC-loads', 'LLC-load-misses',
        'avg_time_sec'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate Miss Rates
    df['L1-miss-rate'] = (df['L1-dcache-load-misses']
                         .div(df['L1-dcache-loads'])
                         .replace([np.inf, -np.inf], np.nan)
                         .fillna(0))
    df['L2-miss-rate'] = (df['L2-load-misses']
                         .div(df['L2-loads'])
                         .replace([np.inf, -np.inf], np.nan)
                         .fillna(0))
    df['LLC-miss-rate'] = (df['LLC-load-misses']
                         .div(df['LLC-loads'])
                         .replace([np.inf, -np.inf], np.nan)
                         .fillna(0))
    
    # Calculate IPC (Instructions Per Cycle)
    df['IPC'] = (df['instructions']
                 .div(df['cycles'])
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0))

    # Categorize benchmarks
    df['op_type'] = df['benchmark_name'].apply(lambda x: 'add' if '_add' in x else 'mul')
    df['is_parallel'] = df['benchmark_name'].str.contains('parallel')
    df['is_transposed'] = df['benchmark_name'].str.contains('transposed')
    
    # Get the "base" implementation name
    df['base_impl'] = df['benchmark_name'].str.replace('numba_parallel_', 'numba_')
    df['base_impl'] = df['base_impl'].str.replace('_transposed', '')
    
    return df

def save_plot(fig, title_str):
    """Saves a matplotlib figure to the plot directory."""
    if not SAVE_PLOTS:
        return
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    
    # Create a safe filename
    filename = title_str.lower().replace(' ', '_').replace('=', '').replace(',', '').replace('+', 'plus') + '.png'
    filepath = os.path.join(PLOT_DIR, filename)
    
    try:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")

# --- Plot 1 Function  ---

def plot_1_level_tiling_effect(df, N, op_type, parallel, transposed, title_prefix):
    """
    Plots Cache Miss Rates and Average Runtime vs. Block Size (B1) for 1-level tiling.
    Filters for B2 == 0.
    Adds a horizontal baseline for the naive implementation.
    """
    
    title = f"{title_prefix} (N={N})"
    filename_str = f"1_level_tiling_{title_prefix}_N{N}"
    
    # ---  Find the baseline (naive) implementation for comparison ---
    baseline_filters = (
        (df['base_impl'] == f'numba_naive_{op_type}') &
        (df['N'] == N) &
        (df['is_parallel'] == parallel) &
        (df['is_transposed'] == transposed)
    )
    baseline_run = df[baseline_filters]

    baseline_metrics = {}
    if not baseline_run.empty:
        baseline_metrics = baseline_run.iloc[0] # Get the first (and only) row
    else:
        print(f"Warning: No baseline 'numba_naive_{op_type}' found for {title}. No baseline will be drawn.")
    

    # Filter data for 1-level tiled implementation
    data = df[
        (df['base_impl'] == f'numba_tiled_{op_type}') &
        (df['N'] == N) &
        (df['is_parallel'] == parallel) &
        (df['is_transposed'] == transposed) &
        (df['B1'] > 0) &  # Only include runs where B1 was used
        (df['B2'] == 0)   # CRITICAL: Ensure it's 1-level tiling
    ].sort_values('B1')

    if data.empty:
        print(f"No data for '{title}'. Skipping plot.")
        return

    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Effect of 1-Level Tiling (B1) \n{title}',
                 fontsize=16, y=1.03)
    
    metrics_to_plot = [
        ('L1-miss-rate', 'L1 Miss Rate'),
        ('L2-miss-rate', 'L2 Miss Rate'),
        ('LLC-miss-rate', 'LLC Miss Rate'),
        ('avg_time_sec', 'Average Runtime (s)')
    ]
    
    for i, (ax, (metric, plot_title)) in enumerate(zip(axes.flat, metrics_to_plot)):
        sns.lineplot(data=data, x='B1', y=metric, ax=ax, marker='o', label='Tiled (B1)')
        
        # ---  Add baseline ---
        if metric in baseline_metrics:
            ax.axhline(baseline_metrics[metric], ls='--', color='red', label='Naive Baseline')
            # Add legend only once
            if i == 0:
                 ax.legend()
        
        
        ax.set_title(plot_title)
        ax.set_xlabel('Block Size (B1)')
        ax.set_ylabel(plot_title)
        
        if metric.endswith('-rate'):
            ax.set_ylim(0, max(ax.get_ylim()[1], 0.1))
        elif metric == 'avg_time_sec':
            ax.set_ylabel(f"{plot_title}")
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(fig, filename_str)
    plt.close(fig)

# --- Plot 2 Function  ---

def plot_2_level_tiling_effect(df, N, op_type, parallel, transposed, title_prefix):
    """
    Plots Cache Miss Rates and Average Runtime vs. Block Size (B2) for 2-level tiling.
    - Grouped by B1.
    - Adds horizontal baseline for the naive implementation (black, dashed).
    - Adds horizontal baseline for the 1-level tiled implementation (matched color, dotted).
    """
    
    title = f"{title_prefix} (N={N})"
    filename_str = f"2_level_tiling_{title_prefix}_N{N}"
    
    # ---  Find the naive baseline (naive) implementation for comparison ---
    baseline_filters = (
        (df['base_impl'] == f'numba_naive_{op_type}') &
        (df['N'] == N) &
        (df['is_parallel'] == parallel) &
        (df['is_transposed'] == transposed)
    )
    baseline_run = df[baseline_filters]

    baseline_metrics = {}
    if not baseline_run.empty:
        baseline_metrics = baseline_run.iloc[0] # Get the first (and only) row
    else:
        print(f"Warning: No naive baseline 'numba_naive_{op_type}' found for {title}. No baseline will be drawn.")
    
    # ---  Find the 1-level tile baselines for comparison ---
    l1_baseline_filters = (
        (df['base_impl'] == f'numba_tiled_{op_type}') &
        (df['N'] == N) &
        (df['is_parallel'] == parallel) &
        (df['is_transposed'] == transposed) &
        (df['B2'] == 0) # B2=0 signifies 1-level tiling
    )
    
    l1_baseline_data = df[l1_baseline_filters] # Get all matching 1-level runs
    
    if l1_baseline_data.empty:
        print(f"Warning: No 1-level 'numba_tiled_{op_type}' baselines found for {title}.")
        l1_baseline_runs = pd.DataFrame() # Create empty DF to avoid later errors
    else:
        # Group by B1 and take the mean to handle potential duplicates in the CSV
        l1_baseline_runs = l1_baseline_data.groupby('B1').mean(numeric_only=True) 
    
    # Filter data for 2-level tiled implementation
    data = df[
        (df['base_impl'] == f'numba_tiled2_{op_type}') &
        (df['N'] == N) &
        (df['is_parallel'] == parallel) &
        (df['is_transposed'] == transposed) &
        (df['B1'] > 0) &   # CRITICAL: Ensure B1 > 0
        (df['B2'] > 0)     # CRITICAL: Ensure B2 > 0
    ].sort_values(['B1', 'B2'])

    if data.empty:
        print(f"No data for '{title}'. Skipping plot.")
        return

    # Treat B1 as a categorical variable for the hue
    data['B1 (L2 Block)'] = data['B1'].astype('category')

    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Effect of 2-Level Tiling (B2) Grouped by B1 \n{title}',
                 fontsize=16, y=1.03)
    
    metrics_to_plot = [
        ('L1-miss-rate', 'L1 Miss Rate'),
        ('L2-miss-rate', 'L2 Miss Rate'),
        ('LLC-miss-rate', 'LLC Miss Rate'),
        ('avg_time_sec', 'Average Runtime (s)')
    ]
    
    # ---  Flags to track if baselines were added for the legend ---
    naive_baseline_added = False
    l1_baseline_added = False
    
    # Get the color palette used by seaborn
    unique_b1s = sorted(data['B1 (L2 Block)'].unique())
    palette = sns.color_palette('Set1', n_colors=len(unique_b1s))
    b1_color_map = dict(zip(unique_b1s, palette))

    for ax, (metric, plot_title) in zip(axes.flat, metrics_to_plot):
        
        sns.lineplot(data=data, x='B2', y=metric, hue='B1 (L2 Block)', 
                     ax=ax, marker='o', palette=palette)
        
        # ---  Add naive baseline (black, dashed) ---
        if metric in baseline_metrics:
            ax.axhline(baseline_metrics[metric], ls='--', color='black')
            naive_baseline_added = True # Mark that we added it
            
            # ---  Add 1-level tile baselines (matched color, dotted) ---
            for b1_val, color in b1_color_map.items():
                if b1_val in l1_baseline_runs.index:
                    l1_metric = l1_baseline_runs.loc[b1_val, metric] # This will now be a scalar
                    if pd.notna(l1_metric): # This check will now work
                        ax.axhline(l1_metric, ls=':', color=color)
                        l1_baseline_added = True
            
        ax.set_title(plot_title)
        ax.set_xlabel('Block Size (B2)')
        ax.set_ylabel(plot_title)
        
        if metric.endswith('-rate'):
            ax.set_ylim(0, max(ax.get_ylim()[1], 0.1))
        elif metric == 'avg_time_sec':
            ax.set_ylabel(f"{plot_title}\n(Lower is Better)")
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Add a global legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    
    # Start with B1 handles/labels
    legend_handles = handles
    legend_labels = labels

    # Manually add baseline entries
    if naive_baseline_added:
        naive_handle = Line2D([0], [0], color='black', ls='--', label='Naive Baseline')
        legend_handles.append(naive_handle)
        legend_labels.append('Naive Baseline')

    if l1_baseline_added:
        l1_handle = Line2D([0], [0], color='gray', ls=':', label='1-Level Tiled (B1=x)')
        legend_handles.append(l1_handle)
        legend_labels.append('1-Level Tiled (B1=x)')

    fig.legend(legend_handles, legend_labels, title='Legend', loc='center right', bbox_to_anchor=(1.20, 0.5))
    
    # Hide individual legends
    for ax in axes.flat:
        if ax.get_legend():
            ax.get_legend().remove()
            
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    save_plot(fig, filename_str)
    plt.close(fig)

# --- Plot 3 Function ---

def get_best_run(data, filters):
    """Helper to filter dataframe and find the row with minimum average runtime."""
    subset = data.copy()
    for key, value in filters.items():
        subset = subset[subset[key] == value]
    
    if subset.empty:
        return None
    
    # Find and return the row with the minimum average runtime
    return subset.loc[subset['avg_time_sec'].idxmin()]

def plot_parallel_comparison_bar(df, N):
    """
    Bar chart showing the parallel effect for the *best* run of each category.
    """
    
    categories = [
        # Naive Add
        {'base': 'Add Naive', 'type': 'Serial',   'filters': {'base_impl': 'numba_naive_add', 'is_parallel': False, 'op_type': 'add'}},
        {'base': 'Add Naive', 'type': 'Parallel', 'filters': {'base_impl': 'numba_naive_add', 'is_parallel': True, 'op_type': 'add'}},
        
        # 1-Level Tiled Add
        {'base': 'Add Tiled 1L', 'type': 'Serial',   'filters': {'base_impl': 'numba_tiled_add', 'is_parallel': False, 'op_type': 'add', 'B2': 0}},
        {'base': 'Add Tiled 1L', 'type': 'Parallel', 'filters': {'base_impl': 'numba_tiled_add', 'is_parallel': True, 'op_type': 'add', 'B2': 0}},
        
        # --- Naive Mul ---
        {'base': 'Mul Naive', 'type': 'Serial',   'filters': {'base_impl': 'numba_naive_mul', 'is_parallel': False, 'is_transposed': False, 'op_type': 'mul'}},
        {'base': 'Mul Naive', 'type': 'Parallel', 'filters': {'base_impl': 'numba_naive_mul', 'is_parallel': True, 'is_transposed': False, 'op_type': 'mul'}},
        
        # --- Naive Mul Transposed ---
        {'base': 'Mul-T Naive', 'type': 'Serial',   'filters': {'base_impl': 'numba_naive_mul', 'is_parallel': False, 'is_transposed': True, 'op_type': 'mul'}},
        {'base': 'Mul-T Naive', 'type': 'Parallel', 'filters': {'base_impl': 'numba_naive_mul', 'is_parallel': True, 'is_transposed': True, 'op_type': 'mul'}},

        # 1-Level Tiled Mul
        {'base': 'Mul Tiled 1L', 'type': 'Serial',   'filters': {'base_impl': 'numba_tiled_mul', 'is_parallel': False, 'is_transposed': False, 'B2': 0}},
        {'base': 'Mul Tiled 1L', 'type': 'Parallel', 'filters': {'base_impl': 'numba_tiled_mul', 'is_parallel': True, 'is_transposed': False, 'B2': 0}},
        
        # 1-Level Tiled Mul Transposed
        {'base': 'Mul-T Tiled 1L', 'type': 'Serial',   'filters': {'base_impl': 'numba_tiled_mul', 'is_parallel': False, 'is_transposed': True, 'B2': 0}},
        {'base': 'Mul-T Tiled 1L', 'type': 'Parallel', 'filters': {'base_impl': 'numba_tiled_mul', 'is_parallel': True, 'is_transposed': True, 'B2': 0}},
        
        # 2-Level Tiled Mul
        {'base': 'Mul Tiled 2L', 'type': 'Serial',   'filters': {'base_impl': 'numba_tiled2_mul', 'is_parallel': False, 'is_transposed': False, 'B1>': 0, 'B2>': 0}},
        {'base': 'Mul Tiled 2L', 'type': 'Parallel', 'filters': {'base_impl': 'numba_tiled2_mul', 'is_parallel': True, 'is_transposed': False, 'B1>': 0, 'B2>': 0}},
        
        # 2-Level Tiled Mul Transposed
        {'base': 'Mul-T Tiled 2L', 'type': 'Serial',   'filters': {'base_impl': 'numba_tiled2_mul', 'is_parallel': False, 'is_transposed': True, 'B1>': 0, 'B2>': 0}},
        {'base': 'Mul-T Tiled 2L', 'type': 'Parallel', 'filters': {'base_impl': 'numba_tiled2_mul', 'is_parallel': True, 'is_transposed': True, 'B1>': 0, 'B2>': 0}},
    ]

    plot_data = []
    
    # Filter for N
    df_n = df[df['N'] == N]

    for cat in categories:
        filters = cat['filters'].copy()
        
        # Handle B > 0 filters
        data_subset = df_n.copy()
        if filters.pop('B1>', 0) > 0: data_subset = data_subset[data_subset['B1'] > 0]
        if filters.pop('B2>', 0) > 0: data_subset = data_subset[data_subset['B2'] > 0]
            
        best_run = get_best_run(data_subset, filters)
        
        if best_run is not None:
            plot_data.append({
                'Base Task': cat['base'],
                'Type': cat['type'],
                'avg_time_sec': best_run['avg_time_sec']
            })
        else:
            print(f"No data for parallel comparison: {cat['base']} {cat['type']}")

    if not plot_data:
        print("No data found for parallel comparison plot. Skipping.")
        return

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=plot_df, x='Base Task', y='avg_time_sec', hue='Type', ax=ax)
    
    ax.set_title(f'Parallel vs. Serial Performance (Best Run, N={N})')
    ax.set_ylabel('Average Time (s) (Lower is Better)')
    ax.set_xlabel('Implementation')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    
    plt.tight_layout()
    save_plot(fig, f'3_parallel_comparison_N{N}')
    plt.close(fig)

# --- Plot 4 Function ---

def plot_transpose_comparison_bar(df, N):
    """
    Bar chart showing the transpose effect for the *best* run of each category.
    """
    
    categories = [
        # --- Naive ---
        {'base': 'Naive Serial', 'access': 'Standard',   'filters': {'base_impl': 'numba_naive_mul', 'is_parallel': False, 'is_transposed': False}},
        {'base': 'Naive Serial', 'access': 'Transposed', 'filters': {'base_impl': 'numba_naive_mul', 'is_parallel': False, 'is_transposed': True}},
        {'base': 'Naive Parallel', 'access': 'Standard',   'filters': {'base_impl': 'numba_naive_mul', 'is_parallel': True, 'is_transposed': False}},
        {'base': 'Naive Parallel', 'access': 'Transposed', 'filters': {'base_impl': 'numba_naive_mul', 'is_parallel': True, 'is_transposed': True}},

        # 1-Level Tiled Serial
        {'base': 'Tiled 1L Serial', 'access': 'Standard',   'filters': {'base_impl': 'numba_tiled_mul', 'is_parallel': False, 'is_transposed': False, 'B2': 0}},
        {'base': 'Tiled 1L Serial', 'access': 'Transposed', 'filters': {'base_impl': 'numba_tiled_mul', 'is_parallel': False, 'is_transposed': True, 'B2': 0}},
        # 1-Level Tiled Parallel
        {'base': 'Tiled 1L Parallel', 'access': 'Standard',   'filters': {'base_impl': 'numba_tiled_mul', 'is_parallel': True, 'is_transposed': False, 'B2': 0}},
        {'base': 'Tiled 1L Parallel', 'access': 'Transposed', 'filters': {'base_impl': 'numba_tiled_mul', 'is_parallel': True, 'is_transposed': True, 'B2': 0}},
        # 2-Level Tiled Serial
        {'base': 'Tiled 2L Serial', 'access': 'Standard',   'filters': {'base_impl': 'numba_tiled2_mul', 'is_parallel': False, 'is_transposed': False, 'B1>': 0, 'B2>': 0}},
        {'base': 'Tiled 2L Serial', 'access': 'Transposed', 'filters': {'base_impl': 'numba_tiled2_mul', 'is_parallel': False, 'is_transposed': True, 'B1>': 0, 'B2>': 0}},
        # 2-Level Tiled Parallel
        {'base': 'Tiled 2L Parallel', 'access': 'Standard',   'filters': {'base_impl': 'numba_tiled2_mul', 'is_parallel': True, 'is_transposed': False, 'B1>': 0, 'B2>': 0}},
        {'base': 'Tiled 2L Parallel', 'access': 'Transposed', 'filters': {'base_impl': 'numba_tiled2_mul', 'is_parallel': True, 'is_transposed': True, 'B1>': 0, 'B2>': 0}},
    ]

    plot_data = []
    
    # Filter for N
    df_n = df[df['N'] == N]

    for cat in categories:
        filters = cat['filters'].copy()
        
        # Handle B > 0 filters
        data_subset = df_n.copy()
        if filters.pop('B1>', 0) > 0: data_subset = data_subset[data_subset['B1'] > 0]
        if filters.pop('B2>', 0) > 0: data_subset = data_subset[data_subset['B2'] > 0]
            
        best_run = get_best_run(data_subset, filters)
        
        if best_run is not None:
            plot_data.append({
                'Base Task': cat['base'],
                'Access Pattern': cat['access'],
                'avg_time_sec': best_run['avg_time_sec']
            })
        else:
            print(f"No data for transpose comparison: {cat['base']} {cat['access']}")

    if not plot_data:
        print("No data found for transpose comparison plot. Skipping.")
        return

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=plot_df, x='Base Task', y='avg_time_sec', hue='Access Pattern', ax=ax)
    
    ax.set_title(f'Standard vs. Transposed Access Performance (Best Run, N={N})')
    ax.set_ylabel('Average Time (s) (Lower is Better)')
    ax.set_xlabel('Implementation')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    save_plot(fig, f'4_transpose_comparison_N{N}')
    plt.close(fig)

# --- Main Execution ---

def main():
    """
    Main function to load data, process it, and generate all plots.
    """
    # Set plot style
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    
    # --- 1. Load Data ---
    try:
        raw_df = pd.read_csv(CSV_FILENAME)
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILENAME}' was not found.")
        print(f"Please place it in the same directory as this script,")
        print("or update the 'CSV_FILENAME' variable.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
        
    print(f"Successfully loaded '{CSV_FILENAME}'.")
    
    # --- 2. Process Data ---
    df = process_dataframe(raw_df)
    print("Data processing complete. Calculated helper columns.")

    # --- 3. Determine N to Plot ---
    if DEFAULT_N_TO_PLOT:
        N_to_plot = DEFAULT_N_TO_PLOT
        if N_to_plot not in df['N'].unique():
            print(f"Warning: N={N_to_plot} not found in data. Using largest N instead.")
            N_to_plot = df['N'].max()
    else:
        N_to_plot = df['N'].max()
        
    print(f"Generating all plots for matrix size N = {N_to_plot}.")
    print(f"Plots will be saved to '{PLOT_DIR}'")

    # --- 4. Generate Plots ---
    
    print("\n--- Generating Plot 1 (1-Level Tiling) ---")
    plot_1_level_tiling_effect(df, N_to_plot, 'add', parallel=False, transposed=False, title_prefix="Add Serial")
    plot_1_level_tiling_effect(df, N_to_plot, 'add', parallel=True, transposed=False, title_prefix="Add Parallel")
    plot_1_level_tiling_effect(df, N_to_plot, 'mul', parallel=False, transposed=False, title_prefix="Mul Serial")
    plot_1_level_tiling_effect(df, N_to_plot, 'mul', parallel=True, transposed=False, title_prefix="Mul Parallel")
    plot_1_level_tiling_effect(df, N_to_plot, 'mul', parallel=False, transposed=True, title_prefix="Mul+Transpose Serial")
    plot_1_level_tiling_effect(df, N_to_plot, 'mul', parallel=True, transposed=True, title_prefix="Mul+Transpose Parallel")

    print("\n--- Generating Plot 2 (2-Level Tiling) ---")
    # Note: 2-level tiling for 'add' was not in the original benchmark script,
    # but the plot function will just skip if it finds no data.
    plot_2_level_tiling_effect(df, N_to_plot, 'add', parallel=False, transposed=False, title_prefix="Add Serial 2L")
    plot_2_level_tiling_effect(df, N_to_plot, 'mul', parallel=False, transposed=False, title_prefix="Mul Serial 2L")
    plot_2_level_tiling_effect(df, N_to_plot, 'mul', parallel=True, transposed=False, title_prefix="Mul Parallel 2L")
    plot_2_level_tiling_effect(df, N_to_plot, 'mul', parallel=False, transposed=True, title_prefix="Mul+Transpose Serial 2L")
    plot_2_level_tiling_effect(df, N_to_plot, 'mul', parallel=True, transposed=True, title_prefix="Mul+Transpose Parallel 2L")

    print("\n--- Generating Plot 3 (Parallel Comparison) ---")
    plot_parallel_comparison_bar(df, N_to_plot)
    
    print("\n--- Generating Plot 4 (Transpose Comparison) ---")
    plot_transpose_comparison_bar(df, N_to_plot)

    print(f"\nAll plots saved to '{PLOT_DIR}' directory.")

    # get the lowest time config for matmul at size N_to_plot but not numpy_mul
    matmul_df = df[(df['op_type'] == 'mul') & (df['N'] == N_to_plot) & (df['base_impl'] != 'numpy_mul')]
    best_row = matmul_df.loc[matmul_df['avg_time_sec'].idxmin()]
    print(f"\nBest MatMul Configuration at N={N_to_plot}:")
    print(best_row[['benchmark_name', 'B1', 'B2', 'is_parallel', 'is_transposed', 'avg_time_sec']])   

if __name__ == "__main__":
    main()

