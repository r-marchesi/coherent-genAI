import pandas as pd

def summarize_results(file_path):
    """
    Reads experiment results from a CSV, calculates summary statistics,
    and saves them to a new CSV file.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the summary statistics.
        Returns None if the file cannot be read.
    """
    try:
        # Read the data from the provided CSV file.
        # This assumes the CSV has columns like 'test_condition', 'test_type',
        # 'c_index'
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    
    # Drop samples where 'test_type' is 'imputed_knn' or 'imputed_mean'
    df = df[~df['test_type'].isin(['imputed_knn', 'imputed_mean'])] 

    # Define the metrics we want to calculate statistics for.
    metrics = ['c_index']

    # Group the DataFrame by 'test_condition' and 'test_type'.
    # For each group, calculate the mean and standard deviation for our metrics.
    summary_stats = df.groupby(['test_condition', 'test_type'])[metrics].agg(['mean', 'std']).reset_index()

    # The aggregation creates multi-level column names.
    # We'll flatten them into a single-level structure for easier access (e.g., 'balanced_accuracy_mean').
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]

    # Clean up the column names by removing any trailing underscores from the grouping columns.
    summary_stats = summary_stats.rename(columns={
        'test_condition_': 'test_condition',
        'test_type_': 'test_type'
    })

    # Define the output file name.
    output_filename = '../../results/downstream/task_06_imputing_test_set_surv/summary_statistics.csv'

    try:
        # Save the resulting summary DataFrame to a new CSV file.
        # We set index=False to avoid writing the DataFrame index as a column.
        summary_stats.to_csv(output_filename, index=False)
        print(f"Successfully created summary file: '{output_filename}'")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
        return None

    return summary_stats

if __name__ == '__main__':
    # Specify the name of your input CSV file.
    input_file = '../../results/downstream/task_06_imputing_test_set_surv/all_imputations_results_long_rf.csv' 

    # Call the function to perform the analysis.
    summary_df = summarize_results(input_file)

    # If the summary was created successfully, print it to the console.
    if summary_df is not None:
        print("\n--- Summary Statistics ---")
        with pd.option_context('display.precision', 3):
            print(summary_df)
