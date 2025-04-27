import pandas as pd
from tqdm import tqdm


def sample_and_save_csv_to_parquet(
    input_csv_path: str, output_parquet_path: str, sample_percentage: float
):
    """
    Samples a percentage of data from a CSV file and saves it as a Parquet file.

    Parameters:
    - input_csv_path (str): The path to the input CSV file.
    - output_parquet_path (str): The path where the output Parquet file will be saved.
    - sample_percentage (float): The percentage of the data to sample (between 0 and 1).
    """
    # Load the CSV file in chunks, removing potential quotes from headers
    iterator = pd.read_csv(input_csv_path, chunksize=10000)
    pbar = tqdm(desc="Reading CSV", unit=" lines", position=0)

    chunks = []
    for chunk in iterator:
        # Strip quotes from column names if present
        chunk.columns = [col.replace("'", "").replace('"', "") for col in chunk.columns]
        chunks.append(chunk)
        pbar.update(len(chunk))

    # Concatenate all chunks into one DataFrame
    df = pd.concat(chunks)
    pbar.close()

    # Sample the data
    sampled_df = df.sample(frac=sample_percentage)

    # Save the sampled data to a Parquet file
    sampled_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    print(
        f"Saved sampled data to {output_parquet_path} containing {len(sampled_df)} rows."
    )


if __name__ == "__main__":
    # Example usage
    input_csv = "./benign_2312_anonymized_HTML.csv"  # Replace with your CSV file path
    output_parquet = "./benign_2312_anonymized_HTML.parquet"  # Replace with your desired Parquet file path
    sample_percent = 0.01  # Sample 1% of the data

    # Call the function
    sample_and_save_csv_to_parquet(input_csv, output_parquet, sample_percent)
