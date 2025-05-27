import json

import pytest

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from sklearn.linear_model import LogisticRegression

import xarray as xr

from prr.utils import (
    convert_to_categorical, get_object_from_module,
    make_feature_combination_array,
    make_feature_combination_score_array,
    get_current_data_version_folder,
    scale_up_dataset,
    is_monotonically_increasing,
    is_monotonically_non_decreasing,
    GridSegment, generate_log_spaced_grid
)

def test_convert_to_categorical():
    df = pd.DataFrame(dict(
        gender=[0, 1, 1], occupation=[1, 0, 0], default=[0, 0, 1]
    ))
    categorical_values_dict = dict(
        gender=[0, 1], occupation=[0, 1], default=[0, 1]
    )

    expected = pd.DataFrame(
        dict(gender=[0, 1, 1], occupation=[1, 0, 0], default=[0, 0, 1]),
    ).astype({
        'gender': CategoricalDtype(categories=[0, 1], ordered=True),
        'occupation': CategoricalDtype(categories=[0, 1], ordered=True),
        'default': CategoricalDtype(categories=[0, 1], ordered=True)
    })

    res = convert_to_categorical(df, categorical_values_dict)
    pd.testing.assert_frame_equal(res, expected)


def test_load_module_object():
    parent_module_name = 'sklearn.linear_model'
    object_name = 'LogisticRegression'
    method_name = 'fit'

    an_object = get_object_from_module(parent_module_name, object_name)
    a_method = getattr(an_object, method_name)

    assert type(LogisticRegression.fit) == type(a_method)


@pytest.mark.parametrize(
    'label_mapping_values,expected',
    [
        (
            {
                "gender": [0, 1],
                "occupation": [0, 1],
            },
            np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ])
        ),
        (
            {
                "gender": [-1, 1],
                "occupation": [-1, 1],
            },
            np.array([
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1]
            ])
        )
    ]
)
def test_make_feature_combination_array(label_mapping_values, expected):
    res = make_feature_combination_array(label_mapping_values)
    np.testing.assert_array_equal(res, expected)


def test_make_feature_combination_scores_array():
    feature_combinations = pd.DataFrame(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], columns=('first', 'second')
    )
    scores = pd.Series([-0.5, -0.25, 0.25, 0.5])

    expected = xr.DataArray(
        [
            [-0.5, -0.25],
            [0.25, 0.5]
        ],
        dims=('first', 'second'),
        coords=dict(first=[0, 1], second=[0, 1])
    )

    res = make_feature_combination_score_array(feature_combinations, scores)

    xr.testing.assert_equal(res, expected)



@pytest.mark.parametrize(
    'folder_names,expected_folder_name',
    [
        (
            ['2023-08-29_09-56-29', '2023-08-29_09-56-30'],
            '2023-08-29_09-56-30'
        ),
        (
            ['2023-08-29_09-56-29', '2023-01-29_09-56-30'],
            '2023-08-29_09-56-29'
        )
    ]
)
def test_get_current_data_version_folder(tmp_path, folder_names, expected_folder_name):
    relative_dir = 'simpsons-paradox/data'
    datadir = tmp_path / relative_dir

    for folder_name in folder_names:
        (datadir / folder_name).mkdir(parents=True)

    res = get_current_data_version_folder(datadir)
    expected = datadir / expected_folder_name

    assert res == expected


def test_scale_up_dataset():
    df = pd.DataFrame(dict(bagels=['poppy', 'garlic'], noshes=['knish', 'matzah']))
    scale_factor = 2
    expected = pd.DataFrame(
        dict(bagels=2 * ['poppy', 'garlic'], noshes=2 * ['knish', 'matzah'])
    )

    res = scale_up_dataset(df, scale_factor)

    pd.testing.assert_frame_equal(res, expected)



@pytest.mark.parametrize(
    'Cs,expected',
    [
        (np.array([0., 1., 2.]), True),
        (np.array([42., 42., 42.]), False),
        (np.array([0, -0.0001, -0.00001]), False),
        (np.array([0, -1, 2]), False)
    ]
)
def test_is_monotonically_increasing(Cs, expected):
    res = is_monotonically_increasing(Cs)
    assert res == expected


@pytest.mark.parametrize(
    'Cs,expected',
    [
        (np.array([0., 1., 2.]), True),
        (np.array([42., 42., 42.]), True),
        (np.array([0, -0.0001, -0.00001]), False),
        (np.array([0, -1, 2]), False)
    ]
)
def test_is_monotonically_non_decreasing(Cs, expected):
    res = is_monotonically_non_decreasing(Cs)
    assert res == expected


@pytest.mark.parametrize(
    "grid_config,ExpectedException,expected",
    [
        (
            [GridSegment(start_power=0, stop_power=1, n_grid=5)],
            None,
            np.logspace(0, 1, 5, endpoint=False)
        ),
        (
            [
                GridSegment(start_power=0, stop_power=1, n_grid=5),
                GridSegment(start_power=1, stop_power=2, n_grid=5)
            ],
            None,
            np.concatenate([
                np.logspace(0, 1, 5, endpoint=False),
                np.logspace(1, 2, 5, endpoint=False)
            ])
        ),
        (
            [
                GridSegment(start_power=1, stop_power=0, n_grid=5)
            ],
            ValueError,
            None
        ),
        (
            [
                GridSegment(start_power=0, stop_power=1, n_grid=5),
                GridSegment(start_power=2, stop_power=1, n_grid=5)
            ],
            ValueError,
            None
        ),
                    (
            [
                GridSegment(start_power=0, stop_power=2, n_grid=5),
                GridSegment(start_power=1, stop_power=1, n_grid=5)
            ],
            ValueError,
            None
        )
    ]
)
def test_generate_log_spaced_grid(grid_config, ExpectedException, expected):
    if ExpectedException:
        with pytest.raises(ExpectedException):
            generate_log_spaced_grid(grid_config)
    else:
        res = generate_log_spaced_grid(grid_config)
        np.testing.assert_array_almost_equal(res, expected)
