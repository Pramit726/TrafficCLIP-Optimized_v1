from pathlib import Path

import pandas as pd

from src.utils.utils import export_data


def aggregate_experiment_results(directory_path, output_filename="master_results.csv"):
    path = Path(directory_path)
    all_dfs = []

    # Iterate through all CSV files in the directory
    for csv_file in path.glob("*.csv"):
        # Load the CSV
        df = pd.read_csv(csv_file)

        # Add the 'Model Name' column using the filename (without .csv extension)
        # Example: 'optimized_L5.0_statsTrue.csv' -> 'optimized_L5.0_statsTrue'
        df["Model Name"] = csv_file.stem

        # Reorder columns to put Model Name first for better readability
        cols = ["Model Name"] + [c for c in df.columns if c != "Model Name"]
        df = df[cols]

        all_dfs.append(df)

    if not all_dfs:
        print("No CSV files found in the specified directory.")
        return None

    # Concatenate all dataframes into one
    master_df = pd.concat(all_dfs, ignore_index=True)
    return master_df


results_folder = "./results/metrics/original"
final_results = aggregate_experiment_results(results_folder)

if final_results is not None:
    export_data(
        final_results,
        export_path="./results/table/original",
        name="comp_table_train-org",
    )
