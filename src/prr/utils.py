import logging
import importlib
from itertools import product
from typing import List, Dict
from pathlib import Path
import datetime
import json

import pandas as pd
from pandas.api.types import CategoricalDtype

import xarray as xr
import numpy as np

import yaml


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_to_categorical(
        df: pd.DataFrame, categorical_values_dict: Dict
    ) -> pd.DataFrame:
    data_categories = {}
    for feature_name, feature_values in categorical_values_dict.items():
        data_categories[feature_name] = CategoricalDtype(
            categories=feature_values, ordered=True
        )
    res = df.copy()
    res = res.astype(data_categories)
    return res


def get_object_from_module(parent_module_name: str, object_name: str):
    '''Returns object from specified parent module'''
    parent_module = importlib.import_module(parent_module_name)
    an_object = getattr(parent_module, object_name)
    return an_object


def make_feature_combination_array(label_mapping_values: Dict) -> np.ndarray:
    res = np.array(list(product(*list(label_mapping_values.values()))))
    return res


def make_feature_combination_score_array(
    feature_combinations: pd.DataFrame, scores: pd.Series
) -> xr.DataArray:
    '''TODO refactor, maybe break into smaller functions with tests'''
    coord_names = feature_combinations.columns
    coords = {}
    for coord_name in coord_names:
        feature_vals = list(set(feature_combinations[coord_name]))
        coords[coord_name] = feature_vals

    # Get score array shape
    score_shape = []
    for vals in coords.values():
        score_shape.append(len(vals))

    score_array = xr.DataArray(
        np.zeros(score_shape),
        dims=coords.keys(),
        coords=coords
    )
    for row_idx, feature_series in feature_combinations.iterrows():
        feature_dict = feature_series.to_dict()
        score_array[tuple(feature_dict.values())] = scores[row_idx]

    return score_array


def get_current_data_version_folder(datadir: Path, format="%Y-%m-%d_%H-%M-%S") -> Path:
    folder_names = []
    for p in datadir.iterdir():
        if p.is_dir():
            folder_names.append(p.name)
    most_recent_str = None
    for timestamp_str in folder_names:
        timestamp = datetime.datetime.strptime(timestamp_str, format)
        if (
            most_recent_str is None
            or timestamp > datetime.datetime.strptime(most_recent_str, format)
        ):
            most_recent_str = timestamp_str
    return datadir / most_recent_str


def scale_up_dataset(df: pd.DataFrame, scale_factor: int) -> pd.DataFrame:
    res = pd.concat(scale_factor * [df], axis=0)
    res = res.reset_index(drop=True)
    return res


def generate_log_spaced_array(start_power: int, stop_power:int, num: int) -> np.ndarray:
    # Generate n floats evenly spaced in log scale
    C = np.logspace(start=start_power, stop=stop_power, num=num, endpoint=False)

    return C