import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes import common as com
import numpy as np


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
        self, in_features: DataFrame, use_pretrained=False, scaler_path=None
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

        # Save the scaler
        joblib.dump(scaler, f"scalers/{self.malicious_label}_scaler.joblib")

        return features
