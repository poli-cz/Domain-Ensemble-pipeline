import pandas as pd
from typing import List, Dict


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
