import pandas as pd

def count_parquet_rows(filepath):
    try:
        df = pd.read_parquet(filepath)
        print(f"Number of rows: {len(df)}")
    except Exception as e:
        print(f"Failed to read the Parquet file: {e}")

if __name__ == "__main__":
    filepath = "2405_clftest_malware_filtered.parquet"  # Replace with your actual file path
    count_parquet_rows(filepath)
