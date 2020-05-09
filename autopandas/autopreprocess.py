import pandas as pd
import numpy as np
import sklearn.preprocessing
import warnings
from typing import Union

class Preprocessor:

    def __init__(self):
        self.scaler_choices = ["maxabs", "minmax", "normalize", "robust", "standard"]
        self.col_scaler = dict()
        self.one_hot_cols = []
        self.binary_cols = []
        self.ordinal_cols = dict()
        self.binarize_cols = dict()

    def check_col_name_type(self, col_name: Union[list, tuple, str]) -> bool:
        if isinstance(col_name, str):
            col_name = [col_name]
        for col in col_name:
            if not isinstance(col_name, str):
                raise TypeError("Column name must be string")

    def _scaler(self, choice: str=None, params: dict=None):
        choice = choice.strip().lower()
        if not params:
            params = dict()
        if choice not in self.scaler_choices:
            raise ValueError("Choice must be either {}".format(", ".join(self.scaler_choices)))
        scaler = None
        if choice == "maxabs":
            scaler = sklearn.preprocessing.MaxAbsScaler(**params)
        elif choice == "minmax":
            scaler = sklearn.preprocessing.MinMaxScaler(**params)
        elif choice == "normalize":
            scaler = sklearn.preprocessing.Normalizer(**params)
        elif choice == "robust":
            scaler = sklearn.preprocessing.RobustScaler(**params)
        elif choice == "standard":
            scaler = sklearn.preprocessing.StandardScaler(**params)
        
        return scaler
    
    def set_scaler(self, col_scaler: dict=None) -> None:
        """
        col_scaler: keys should be column names and values should be a dictionary containing
                    keys (choice, params) and values should be (choice of scaler, parameters of scaler)
        """
        for col, dic in col_scaler.items():
            self.check_col_name_type(col)
            if "choice" in dic.keys():
                self.col_scaler[col] = self._scaler(dic["choice"], dic.get("params"))
            else:
                warnings.warn("Scaler choice not specified for {}, skipping".format(col))
    
    def set_one_hot(self, col_names: Union[list, tuple]) -> None:
        """
        col_names: iterable containing names of columns to be one hot encoded
        """
        self.check_col_name_type(col_names)
        self.one_hot_cols = col_names

    def set_binary(self, col_names: Union[list, tuple, dict]) -> None:
        """
        col_names: iterable containing names of columns to be one hot encoded,
                    if dictionary => contains names of columns as keys and dictionaries as values,
                    these dictionaries should have unique values as keys and corresponding values should be 0 or 1
        """
        self.check_col_name_type(col_names)
        self.binary_cols = col_names

    def set_ordinal(self, col_names: Union[list, tuple, dict]) -> None:
        """
        col_names: iterable containing names of columns to be one hot encoded,
                    if dictionary => contains names of columns as keys and dictionaries as values,
                    these dictionaries should have unique values as keys and corresponding values should be integers
        """
        self.check_col_name_type(col_names)
        self.ordinal_cols = col_names

    def set_binarize(self, col_names_threshold: dict) -> None:
        """
        col_names_threshold: keys should be column names and corresponding values should be a threshold
                                where by any values less than or equal to it will be mapped to 0 and 1 otherwise
        """
        for col, threshold in col_names_threshold.items():
            if (not isinstance(col, str)) or (not (isinstance(threshold, int) or isinstance(threshold, float))):
                raise TypeError("Column name must be string and threshold must be either of type integer or float")
        self.binarize_cols = col_names_threshold

    def set_binning(self, col_bins: dict=None) -> None:
        """
        col_bins: keys should be column names and values should be a dictionary containing
                    parameters of sklearn's KBinsDiscretizer
        """
        for col, params in col_bins.items():
            if not isinstance(params, dict):
                params = dict()
            self.check_col_name_type(col)
            self.col_scaler[col] = sklearn.preprocessing.KBinsDiscretizer(**params)