import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
from matplotlib.gridspec import GridSpec


      

def create_summary_plot(
    data: pd.DataFrame, 
    metric: str, 
    title: str, 
    save_path: str, # 
):
    """
    Generates and saves a readable plot for a given metric without displaying it.
    """
    print(f"\n--- Generating and saving plot for: {metric} ---")

    # Set a professional and clean plot style
    sns.set_style("whitegrid")
    sns.set_context("paper")

    # --- Data Preprocessing for Logical Ordering ---
    data['n_removed'] = data['test_condition'].apply(
        lambda x: 0 if x == 'full_data' else x.count('_') + 1
    )
    if 'cancer_label_only' in data['test_condition'].unique():
        data.loc[data['test_condition'] == 'cancer_label_only', 'n_removed'] = data['n_removed'].max() + 1
    
    plot_order = data.sort_values(by=['n_removed', 'test_condition']).test_condition.unique()

    # --- Color Palette Definition ---
    
    custom_palette = {
    'full_data': "#9c2409",
    'ablation': '#e66000',
    'imputed_coherent': '#56b4e9',
    'imputed_multi': '#0072b2'
    }

    # --- Plot Generation ---
    g = sns.catplot(
        data=data, 
        x='test_condition', 
        y=metric, 
        hue='test_type',
        order=plot_order, 
        kind='bar', 
        height=7,
        aspect=2.5, 
        legend_out=True,
        errorbar='sd',
        palette=custom_palette
    )

    # --- Readability and Aesthetic Improvements ---
    g.fig.suptitle(title, y=1.03, fontsize=24, fontweight='bold')
    g.set_axis_labels("Test Condition (Modalities Removed)", f"{metric.replace('_', ' ').title()}", fontsize=16)
    g.set_xticklabels(rotation=45, ha='right', fontsize=12)
    g.set_yticklabels(fontsize=12)
    
    legend = g._legend
    legend.set_title('Test Set', prop={'size': 14, 'weight': 'bold'})
    legend.set_bbox_to_anchor((1.02, 0.5))
    legend.set_loc('center left')
    for t in legend.texts:
        t.set_fontsize(12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Save and Close (No Display) ---
    # The plot is saved to the specified path.
    g.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # Close the figure to free up memory and prevent it from being displayed.
    plt.close(g.fig)



def plot_grouped_bar_chart(df, metric_mean_col, metric_std_col, title, y_label, palette, save_path=None):
    """
    Generates and saves a grouped bar chart for a given metric.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        metric_mean_col (str): The name of the column with the metric's mean values.
        metric_std_col (str): The name of the column with the metric's std values.
        title (str): The title for the plot.
        y_label (str): The label for the y-axis.
        palette (dict): A dictionary mapping test_type to color.
        save_path (str, optional): The file path to save the plot. If None, displays the plot.
    """
    # --- Data and Category Setup ---
    # Get unique categories for grouping and ensure a consistent order
    types = sorted(df['test_type'].unique())
    
    # Manually order the conditions to ensure 'full_data' is first
    all_conditions = list(df['test_condition'].unique())
    if 'full_data' in all_conditions:
        all_conditions.remove('full_data')
        # Sort the remaining conditions alphabetically for consistency
        conditions_ordered = ['full_data'] + sorted(all_conditions)
    else:
        conditions_ordered = sorted(all_conditions)
    
    # Use pandas Categorical type to enforce the desired order for plotting
    df['test_condition'] = pd.Categorical(df['test_condition'], categories=conditions_ordered, ordered=True)
    df = df.sort_values('test_condition')
    # Get the unique conditions again, now in the correct order
    conditions = df['test_condition'].unique()
    
    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))

    n_conditions = len(conditions)
    n_types = len(types)
    
    bar_width = 0.2
    index = np.arange(n_conditions)

    # --- Create Bars for Each Type ---
    for i, test_type in enumerate(types):
        # Pivot the data to handle missing values gracefully
        pivot_df = df[df['test_type'] == test_type].pivot(index='test_condition', columns='test_type')
        
        # Reindex to match the full, ordered list of conditions, filling missing values with 0
        means_data = pivot_df[metric_mean_col].reindex(conditions, fill_value=0)
        stds_data = pivot_df[metric_std_col].reindex(conditions, fill_value=0)
        
        # The reindexed data is a Series, get the values
        means = means_data[test_type].values
        stds = stds_data[test_type].values

        bar_positions = index + i * bar_width - (bar_width * (n_types-1) / 2)
        color = palette.get(test_type)
        ax.bar(bar_positions, means, bar_width, label=test_type, yerr=stds, capsize=4, color=color)

    # --- Final Touches ---
    ax.set_xlabel('Data Condition', fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    # Format x-tick labels for better readability
    formatted_labels = [label.replace('_', ' ').title() for label in conditions]
    ax.set_xticks(index)
    ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=12)
    
    min_val = df[metric_mean_col].min()
    ax.set_ylim([max(0, min_val - 0.1), 0.6])
    
    ax.legend(title='Experiment Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    
    fig.tight_layout(rect=[0, 0, 0.9, 1]) 

    # --- Save or Show Plot ---
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()
    
    plt.close(fig)


# --- Define Custom Palette ---
# Palette keys now match the 'test_type' column in the new CSV data
custom_palette = {
    'full_data': "#9c2409",
    'ablation': '#e66000',
    'imputed_coherent': '#56b4e9',
    'imputed_multi': '#0072b2'
}




results_path = '../../results/downstream/task_05_imputing_test_set'
N_RUNS = 10

# Save the final results to a CSV file
final_results = pd.read_csv(os.path.join(results_path, f'results_{str(N_RUNS)}_runs.csv'))
summary_results = pd.read_csv(os.path.join(results_path, f'summary_statistics.csv'))

# Create a plot for each metric
create_summary_plot(final_results, 'balanced_accuracy', 'Classification Performance on Sparce Data vs Synthetic Data', save_path=os.path.join(results_path, f'balanced_accuracy_{str(N_RUNS)}_runs.png'))
create_summary_plot(final_results, 'macro_f1_score', 'Classification Performance on Sparce Data vs Synthetic Data', save_path=os.path.join(results_path, f'f1_score_plot_{str(N_RUNS)}_runs.png'))

plot_grouped_bar_chart(summary_results.copy(),     
                       metric_mean_col='balanced_accuracy_mean', 
                       metric_std_col='balanced_accuracy_std', 
                       title='Model Performance: Balanced Accuracy by Data Condition and Experiment Type',
                       y_label='Balanced Accuracy', 
                       palette=custom_palette,
                       save_path=os.path.join(results_path, f'final_balanced_accuracy_{str(N_RUNS)}_runs.png'))
plot_grouped_bar_chart(
    summary_results.copy(),
    metric_mean_col='macro_f1_score_mean',
    metric_std_col='macro_f1_score_std',
    title='Model Performance: Macro F1-Score by Data Condition and Experiment Type',
    y_label='Macro F1-Score',
    palette=custom_palette,
    save_path=os.path.join(results_path, f'final_f1_score_{str(N_RUNS)}_runs.png')
)