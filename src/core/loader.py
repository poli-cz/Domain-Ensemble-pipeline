import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes import common as com
import numpy as np
from sklearn.model_selection import train_test_split


import pandas as pd
from typing import List, Dict


# logging


class Segmenter:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy(deep=True)
        self.prefixes = ["dns_", "tls_", "html_", "geo_", "rdap_", "lex_", "ip_"]
        self.subsets: Dict[str, pd.DataFrame] = {}

    def create_base_subsets(
        self, include_label: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Creates a dictionary of dataframes where each dataframe contains only the columns
        that start with a specific prefix and optionally includes the 'label' column.
        """
        for prefix in self.prefixes:
            subset_df = self.df.loc[:, self.df.columns.str.startswith(prefix)]
            if include_label and "label" in self.df.columns:
                subset_df["label"] = self.df["label"].copy()

            self.subsets[prefix] = subset_df

        return self.subsets

    def create_aggregated_subsets(
        self, aggregates: List[List[str]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Creates aggregated subsets based on lists of prefixes. Aggregates columns
        from each group, plus the 'label' column.
        """

        self.aggregates = aggregates
        self.aggregated_subsets = {}
        if not self.aggregates:
            raise ValueError("No aggregates provided for creating subsets.")
        if not isinstance(self.aggregates, list):
            raise TypeError("Aggregates should be a list of lists.")
        if not all(isinstance(group, list) for group in self.aggregates):
            raise TypeError("Each group in aggregates should be a list.")
        if not all(
            all(isinstance(prefix, str) for prefix in group)
            for group in self.aggregates
        ):
            raise TypeError("Each prefix in aggregates should be a string.")

        for group in self.aggregates:
            pattern = "|".join(f"^{prefix}" for prefix in group)
            subset_df = self.df.loc[
                :, self.df.columns.str.contains(pattern) | (self.df.columns == "label")
            ]

            # Remove index column if it exists
            subset_df = subset_df.loc[
                :, ~subset_df.columns.str.contains("^index$", case=False)
            ]

            subset_df.reset_index(drop=True, inplace=True)

            key = "+".join(group) + "agg"
            self.subsets[key] = subset_df
            self.aggregated_subsets[key] = subset_df

        return self.subsets

    def get_subsets(self) -> Dict[str, pd.DataFrame]:
        return self.subsets

    def get_aggregated_subsets(self) -> Dict[str, pd.DataFrame]:
        return self.aggregated_subsets


class Loader:
    def __init__(
        self,
        benign_files,
        malicious_files,
        benign_label="benign",
        malicious_label="malware",
        subsample=1.0,
    ):
        self.benign_files = benign_files
        self.malicious_files = malicious_files
        self.benign_label = benign_label
        self.malicious_label = malicious_label
        self.subsample = subsample

    def get_malicious_label(self):
        return self.malicious_label

    def union_tables(self, tables: [pa.Table]) -> pa.Table:
        union_table = tables[0]
        for table in tables[1:]:
            right_not_in_union = union_table.join(
                right_table=table,
                keys="domain_name",
                join_type="right anti",
                coalesce_keys=True,
                use_threads=True,
            )
            union_table = pa.concat_tables([union_table, right_not_in_union])
        return union_table

    def load_one_parket(self, filename) -> pd.DataFrame:
        # Use malicious schema
        schema = pq.read_table(filename[0]).schema

        test_tables = [pq.read_table(f).cast(schema) for f in filename]

        test_tables = self.union_tables(test_tables).to_pandas()

        df = self.cast_timestamp(test_tables)
        df.fillna(-1, inplace=True)

        return self.scale(
            df, use_pretrained=True, scaler_path="scalers/phishing_scaler.joblib"
        )

    def cast_timestamp(self, df: DataFrame):
        for col in df.columns:
            if com.is_timedelta64_dtype(df[col]):
                df[col] = df[
                    col
                ].dt.total_seconds()  # This converts timedelta to float (seconds)
            elif com.is_datetime64_any_dtype(df[col]):
                df[col] = (
                    df[col].astype(np.int64) // 10**9
                )  # Converts datetime64 to Unix timestamp (seconds)

        return df

    def load(self) -> pd.DataFrame:
        # Use malicious schema
        m_schema = pq.read_table(self.malicious_files[0]).schema
        b_schema = pq.read_table(self.benign_files[0]).schema

        benign_tables = [pq.read_table(f).cast(b_schema) for f in self.benign_files]

        malicious_tables = [
            pq.read_table(f).cast(m_schema) for f in self.malicious_files
        ]

        benign = self.union_tables(benign_tables).to_pandas()
        malicious = self.union_tables(malicious_tables).to_pandas()

        # iterate over both columns and filter columns that are not in both

        benign["label"] = self.benign_label
        malicious["label"] = self.malicious_label

        df = pd.concat([benign, malicious])

        df = self.cast_timestamp(df)
        df.fillna(-1, inplace=True)

        if self.subsample < 1.0:
            df = df.sample(frac=self.subsample, random_state=42)

        if "domain_name" in df.columns:
            df.drop("domain_name", axis=1, inplace=True)

        return df

    def scale(
        self,
        in_features: DataFrame,
        use_pretrained=False,
        scaler_path=None,
        stage=None,
        model=None,
    ) -> DataFrame:
        """
        Scale the features of the DataFrame using the provided scaler.
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        import joblib

        if not use_pretrained:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(in_features)

        else:
            # load scaler
            scaler = joblib.load(scaler_path)
            scaled_data = scaler.transform(in_features)

        features = pd.DataFrame(scaled_data, columns=in_features.columns)

        # Save the scalei

        joblib.dump(
            scaler, f"scalers/{self.malicious_label}_{model}_{stage}_scaler.joblib"
        )

        return features


class DataMannager:
    def __init__(self, loader: Loader, class_map: dict):
        self.loader = loader
        self.malicious_label = loader.get_malicious_label()
        self.class_map = class_map

    def get_df(self) -> pd.DataFrame:
        return self.df

    def load(self):
        self.df = self.loader.load()

    def segment(self, aggregates: list) -> dict:
        if not hasattr(self, "df"):
            raise ValueError("DataFrame not loaded. Call load() first.")

        self.segmenter = Segmenter(self.df)
        print("Segmenter initialized. Creating subsets...")
        self.segmenter.create_base_subsets()  # create base subsets
        self.segmenter.create_aggregated_subsets(aggregates)

        self.subset_dfs = self.segmenter.get_aggregated_subsets()

        return self.subset_dfs

    def save_data(self, base_path: str = "./data/") -> None:
        """
        Save the DataFrame to a parquet file.
        """
        if not hasattr(self, "df"):
            raise ValueError("DataFrame not loaded. Call load() first.")

        # check for self.subset_dfs if not called segment
        if not hasattr(self, "subset_dfs"):
            raise ValueError("Subsets not created. Call segment() first.")

        i = 1

        for prefix, subset_df in self.subset_dfs.items():
            X = subset_df.drop("label", axis=1)
            y = subset_df["label"].map(self.class_map)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42
            )

            dump = {
                "X_train": X_train.to_numpy(),
                "X_test": X_test.to_numpy(),
                "Y_train": y_train.to_numpy(),
                "Y_test": y_test.to_numpy(),
                "columns": X.columns,
            }

            # save using pickle to sata
            import pickle

            with open(
                f"{base_path}whole_split_{i}_{self.malicious_label}.pkl", "wb"
            ) as f:
                pickle.dump(dump, f)

            print(
                f"Saved validation stage {i} to {base_path}whole_split_{i}_{self.malicious_label}.pkl"
            )
            i += 1
