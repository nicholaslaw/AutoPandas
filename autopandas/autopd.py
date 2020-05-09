import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

class AutoPandas:
    def __init__(self, df: pd.DataFrame, ignore_cols: list=None) -> None:
        """
        df: Pandas DataFrame
            raw dataframe
        ignore_cols: list
            contains names of columns to ignore
        """
        self.df = df
        if not ignore_cols:
            ignore_cols = []
        self.ignore_cols = ignore_cols
        self.binary_cols = []
        self.cat_cols = []
        self.num_cols = []
        self.null_cols = []

    def discover_type(self, cat_unique: int = 10) -> None:
        """
        cat_unique: integer
            indicating max number of unique values in a column for it to be considered as a categorical column

        Find bin
        """
        cat_cols = self.remove_ignore_cols(self.df.nunique()[self.df.nunique() <= cat_unique].keys().tolist())
        self.binary_cols = self.remove_ignore_cols(self.df.nunique()[self.df.nunique() == 2].keys().tolist())
        self.cat_cols = list(set(cat_cols) - set(self.bin_cols))
        self.num_cols = self.remove_ignore_cols(list(set(self.df.columns) - set(self.cat_cols) - set(self.bin_cols)))

    def remove_ignore_cols(self, cols: list) -> list:
        """
        cols: list
            Each element is the name of a column in the dataframe

        Remove ignored columns from the list input
        """
        return list(set(cols) - set(self.ignore_cols))

    def discover_null(self) -> pd.DataFrame:
        """
        Assign attribute .null_cols to a dataframe containing names of columns with null values
        """
        null_counts = self.df.isna().sum()
        null_df = pd.DataFrame({'Column': null_counts.index, 'Null Counts': null_counts.values})
        null_df = null_df.loc[null_df["Null Counts"] != 0].reset_index().drop("index", axis=1)
        self.null_cols = null_df["Column"].tolist()
        return null_df

    def plot_scatter_matrix(self, hue: str=None) -> None:
        """
        hue: str
            indicating hue for plot

        generate pair plots for numerical variables
        """
        if hue not in self.df.columns:
            warnings.warn("{} is not found in dataframe's columns, ignoring hue...".format(hue))
            hue = None
        if hue:
            sns.pairplot(self.df[self.num_cols])
        else:
            sns.pairplot(self.df[self.num_cols + [hue]], hue=hue)

    def plot_dist(self) -> None:
        """
        plots distribution of all numerical columns
        """
        for col in self.num_cols:
            plt.figure()
            sns.distplot(df[col].tolist())

    def plot_counts(self, hue: str=None) -> None:
        plot_cols = self.binary_cols + self.cat_cols
        if hue not in self.df.columns:
            warnings.warn("{} is not found in dataframe's columns, ignoring hue...".format(hue))
            hue = None
        for col in plot_cols:
            plt.figure()
            sns.countplot(x=col, data=self.df, hue=hue)