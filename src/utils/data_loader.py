import pandas as pd
import glob
import os


def load_or_merge_csvs(raw_path="../data/raw", processed_path="../data/processed/merged_tweets.csv"):
    """
    Load merged dataset if it exists, otherwise merge raw CSVs and save the merged file.

    :param raw_path: Path to the folder containing raw CSVs.
    :param processed_path: Path to save/load the merged dataset.
    :return: Pandas DataFrame with merged data.
    """
    # Check if merged file already exists
    if os.path.exists(processed_path):
        print(f"Loading merged dataset from {processed_path}")
        return pd.read_csv(processed_path)

    # If not, merge all CSVs
    print(f"Merging CSV files from {raw_path}...")
    csv_files = glob.glob(f"{raw_path}/*.csv")

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Save the merged file
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    merged_df.to_csv(processed_path, index=False)
    print(f"Saved merged dataset to {processed_path}")

    return merged_df
