from tensorflow.python.client import device_lib
import torch

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes import common as com
from pyarrow import Table
from pycaret.utils import version
from pycaret.classification import *

for device in device_lib.list_local_devices():
    print(device.physical_device_desc)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def union_tables(tables: [pa.Table]) -> pa.Table:
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


# #############################################################
# EDIT this to specify benign / malicious datasets to use     #
# #############################################################
benign_dataset_filenames = [
    "parkets/benign_2312.parquet",
]
malicious_dataset_filenames = ["parkets/phishing_2406_strict.parquet"]
# #############################################################
# EDIT this for to set appropriate labels (malware, dga, ...) #
# #############################################################
benign_label = "benign"
malicious_label = "phishing"
# #############################################################

# Unify malicious datasets and benign datasets
schema = (
    pq.read_table(malicious_dataset_filenames[0])
).schema  # Use the schema from the first malicious filename
benign_tables = [
    pq.read_table(filename).cast(schema) for filename in benign_dataset_filenames
]
malicious_tables = [
    pq.read_table(filename).cast(schema) for filename in malicious_dataset_filenames
]
malicious = union_tables(malicious_tables)
benign = union_tables(benign_tables)

# Convert pyarrow tables to pandas dataframes
df_benign = benign.to_pandas()
df_malicious = malicious.to_pandas()

# Set appropriate labels
df_benign["label"] = benign_label
df_malicious["label"] = malicious_label
class_map = {benign_label: 0, malicious_label: 1}


# ===================
# AUTO BALANCING !!!
# Subsample benign to match the size of malicious
# df_benign = df_benign.sample(n=len(df_malicious))
# ===================

# Concatentate benign and malicious
df = pd.concat([df_benign, df_malicious])


def cast_timestamp(df: DataFrame):
    """
    Cast timestamp fields to seconds since epoch.
    """
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


df = cast_timestamp(df)

# Handle NaNs
df.fillna(-1, inplace=True)


# SUBSAMPLE1 (OPTIONAL)
subsample = 1.00  # 1.0 means no subsample
if subsample < 1.0:
    df = df.sample(frac=subsample)

# Drop the domain name column
df.drop("domain_name", axis=1, inplace=True)


labels = df["label"].apply(lambda x: class_map[x])  # y vector
features = df.drop("label", axis=1).copy()  # X matrix


print(f"Total features after augmentation: {features.shape[1]}")

print(f"Total samples: {len(df)}")
print(f"Benign count: {len(df_benign)}")
print(f"Malicious count: {len(df_malicious)}")


# pycaret setup and run

version()


clf1 = setup(
    df,
    target="label",
    session_id=51,
    log_experiment=False,
    experiment_name="feta1",
    index=False,
)

best_model = compare_models()
