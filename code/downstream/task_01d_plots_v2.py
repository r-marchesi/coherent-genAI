import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def _plot_bars_polished(ax, df_to_plot, metric_base, data_type_order, color_palette, legend_labels):
    """
    Helper function to draw a polished grouped bar chart on a given matplotlib axis.
    """
    metric_mean_col = f'{metric_base}_mean'
    metric_std_col = f'{metric_base}_std'
    
    # --- Data Preparation ---
    pivot_df_mean = df_to_plot.pivot(index='modality', columns='data_type', values=metric_mean_col)
    pivot_df_std = df_to_plot.pivot(index='modality', columns='data_type', values=metric_std_col)
    
    pivot_df_mean = pivot_df_mean[data_type_order]
    pivot_df_std = pivot_df_std[data_type_order]

    modalities = pivot_df_mean.index
    n_modalities = len(modalities)
    n_bars = len(data_type_order)
    
    x = np.arange(n_modalities)
    width = 0.2
    
    # --- Bar Plotting ---
    for i, data_type in enumerate(data_type_order):
        # Center the group of bars around the tick
        position = x - (width * n_bars / 2) + (i * width) + (width / 2)
        means = pivot_df_mean[data_type]
        stds = pivot_df_std[data_type]
        
        ax.bar(position, means, width, 
               label=legend_labels[data_type],
               yerr=stds, 
               capsize=0,
               color=color_palette[data_type],
               zorder=10)
    
    # --- Aesthetic Refinements ---
    ax.yaxis.grid(True, linewidth=0.7, color="#CDCCCC", zorder=0)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#B0B0B0')

    ax.tick_params(axis='x', colors='#505050', length=0)
    ax.tick_params(axis='y', colors='#505050')

    ax.set_xlabel("Data Modality", fontsize=16, labelpad=15, color='#303030')
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, rotation=0, fontsize=16)
    
    ax.set_ylim(bottom=0, top=1)

def create_polished_plots(csv_path, output_dir):
    """
    Main function to generate both individual and combined performance plots
    with a final, polished, and compact aesthetic.
    """
    # --- 1. Setup ---
    if not os.path.exists(csv_path):
        print(f"Error: The file was not found at {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-white')
    df = pd.read_csv(csv_path)
    
    tasks = df['task'].unique()
    metrics = ['balanced_accuracy', 'f1_macro']
    data_type_order = ['real', 'synthetic_from_coherent', 'synthetic_from_multi']
    
    color_palette = {
        'real': '#e66000',
        'synthetic_from_coherent': '#56b4e9',
        'synthetic_from_multi': '#0072b2'
    }
    legend_labels = {
        'real': 'Real',
        'synthetic_from_coherent': 'Generated\n(Coherent Denoising)',
        'synthetic_from_multi': 'Generated\n(Multi-condition)'
    }

    # --- NEW: Dictionary to map metric names to desired Y-axis labels ---
    ylabel_map = {
        'balanced_accuracy': 'Balanced Accuracy',
        'f1_macro': 'F1 Score'
    }

    # --- Part A: Generate 4 Individual Plots ---
    print("--- Generating 4 individual polished plots ---")
    for task in tasks:
        for metric_base in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            task_df = df[df['task'] == task].copy()
            _plot_bars_polished(ax, task_df, metric_base, data_type_order, color_palette, legend_labels)
            
            # Use the new map to set the y-label
            ax.set_ylabel(ylabel_map.get(metric_base, metric_base), fontsize=16, color='#303030')
            
            ax.set_title(f"Performance for '{task.replace('_', ' ').title()}' Task", fontsize=16, pad=20, weight='bold')
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, title="Test Data Origin", fontsize=16, title_fontsize=18,
                       loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
            fig.subplots_adjust(left=0.1, right=0.80, top=0.9, bottom=0.15)
            plot_filename = f"{task}_{metric_base}_performance.png"
            output_path = os.path.join(output_dir, plot_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    # --- Part B: Generate 2 Combined Plots with Side-by-Side Panels ---
    print("\n--- Generating 2 combined polished plots ---")
    for metric_base in metrics:
        print(f"Generating combined plot for Metric: '{metric_base}'...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
        
        for i, task in enumerate(tasks):
            ax = axes[i]
            task_df = df[df['task'] == task].copy()
            _plot_bars_polished(ax, task_df, metric_base, data_type_order, color_palette, legend_labels)
            ax.set_title(f"Task: {task.replace('_', ' ').title()}", fontsize=18, weight='bold', pad=15)
            
        # Use the new map to set the y-label
        axes[0].set_ylabel(ylabel_map.get(metric_base, metric_base), fontsize=16, color='#303030')
        axes[0].tick_params(axis='y', which='major', labelsize=14)
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Test Data Origin", fontsize=12, title_fontsize=14, labelspacing=1.2,
                   loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
        
        fig.suptitle(f"Comparing Classifiers Performance on Real vs Synthetic Data", fontsize=20, weight='bold')
        
        fig.subplots_adjust(left=0.08, right=0.95, wspace=0.18, top=0.80, bottom=0.15)
        
        plot_filename = f"combined_{metric_base}_performance.png"
        output_path = os.path.join(output_dir, plot_filename)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        print(f"  -> Saved combined plot to {output_path}")

# ... (your __main__ block remains the same) ...
if __name__ == '__main__':
    csv_file_path = '../../results/downstream/task_01_train_on_real/summary_results_10_runs.csv'
    plots_output_directory = '../../results/downstream/task_01_train_on_real/images_10_runs/'
    
    create_polished_plots(csv_file_path, plots_output_directory)