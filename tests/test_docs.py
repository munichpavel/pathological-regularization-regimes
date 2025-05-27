"""Tests for doc utilities."""
# from pathlib import Path
# import os
import json
from math import exp

import numpy as np

from omegaconf import OmegaConf
import pytest

import yaml

from sklearn.linear_model import LogisticRegression

import pandas as pd

from prr.docs import (
    find_most_adversarial_c,
    extract_discussion_config_params,
    get_multirun_artifacts,
    make_trend_grouped_df, get_multirun_reports,
    classify_dataset_runs_adversariality
)

from prr.utils import make_feature_combination_array, make_feature_combination_score_array
from prr.trends import make_a_trend_report, make_trend_reports


def test_find_most_adversarial_c():
    cs = np.array([0.1, 1., 10.])
    trends = np.array([-1., 0., 0.5])
    expected = dict(C=0.1, trend=-1)

    res = find_most_adversarial_c(cs, trends)
    assert res == expected


def test_extract_discussion_config_params():
    conf_yml_str = """data_version_folder: simpsons-paradox/data/2023-11-24-manual
clf:
  parent_module: sklearn.linear_model
  class_name: LogisticRegression
  kwargs:
    C: 0.6
    fit_intercept: true
    solver: lbfgs
    penalty: l2
encoding:
  label_mapping_values:
    gender:
    - 0
    - 1
    occupation:
    - 0
    - 1"""


    conf = OmegaConf.create(conf_yml_str)
    res = extract_discussion_config_params(a_config=conf)
    expected = dict(
        algorithm_class_name='LogisticRegression',
        fit_intercept=True,
        solver='lbfgs'
    )

    assert res == expected


def test_get_multirun_json_artifacts(tmp_path):
    multirun_dir = tmp_path / 'multirun-hh-mm-ss'
    multirun_dir.mkdir()
    n_runs = 10
    artifact_filenames = ['report.json']
    for run_idx in range(n_runs):
        json_data = {'idx': run_idx}
        run_dir = multirun_dir / str(run_idx)
        run_dir.mkdir()
        for artifact_filename in artifact_filenames:
            with open(run_dir / artifact_filename, 'w') as fp:
                json.dump(json_data, fp=fp)

    multirun_artifacts = get_multirun_artifacts(
        multirun_dir, artifact_filenames=artifact_filenames
    )

    values = [run_artifact[artifact_filenames[0]]['idx'] for run_artifact in multirun_artifacts]
    assert len(values) == n_runs and set(values) == set(range(n_runs))


def test_get_multirun_yaml_artifacts(tmp_path):
    """Same as above but with yaml rather than json"""
    multirun_dir = tmp_path / 'multirun-hh-mm-ss'
    multirun_dir.mkdir()
    n_runs = 10
    artifact_filenames = ['config.yaml']

    for run_idx in range(n_runs):
        config = {'idx': run_idx}
        run_dir = multirun_dir / str(run_idx)
        run_dir.mkdir()
        for artifact_filename in artifact_filenames:
            with open(run_dir / artifact_filename, 'w') as fp:
                yaml.dump(config, fp)

    multirun_artifacts = get_multirun_artifacts(
        multirun_dir, artifact_filenames=artifact_filenames
    )

    values = [run_artifact[artifact_filenames[0]]['idx'] for run_artifact in multirun_artifacts]
    assert len(values) == n_runs and set(values) == set(range(n_runs))


def binary_logistic_regression_scores_via_params(b: np.ndarray, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Auxiliary function for test(s) below"""
    model = LogisticRegression()
    model.coef_ = weights
    model.intercept_ = b
    model.classes_ = np.array([0, 1])

    # Compute model probabilities
    res = model.predict_proba(X)[:, 1]
    return res


@pytest.mark.parametrize(
    "b, weights, X, expected_probs",
    [
        (np.array([0.0]), np.array([[0.0]]), np.array([[0], [1]]), np.array([1 / 2., 1 / 2.])),
        (np.array([0.0]), np.array([[1.0]]), np.array([[0], [1]]), np.array([1 / 2., 1 / (1 + exp(-1.))])),
        (np.array([0.0]), np.array([[1.0, -1.0]]), np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([1 / 2., 1/(1 + exp(1)), 1/(1 + exp(-1)), 1/2.])),
        (np.array([1.0]), np.array([[1.0, 1.0]]), np.array([[1, 1]]), np.array([1 / (1 + exp(-3))])),
    ]
)
def test_binary_logistic_regression_scores_via_params(b, weights, X, expected_probs):
    res = binary_logistic_regression_scores_via_params(b, weights, X)
    assert np.allclose(res, expected_probs, atol=1e-6), f"Expected {expected_probs}, got {res}"


def test_make_trend_grouped_df():
    non_trend_run_artifacts = [
        {
            'coefs.json': {'b': [0], 'w': [[0., 1.]]},
            '.hydra/config.yaml': {
                'clf': {'kwargs': {'C': 0.1}},
                'data_version_folder': 'dataset_1'
            }
        },
        {
            'coefs.json': {'b': [0], 'w': [[1., -1.]]},
            '.hydra/config.yaml': {
                'clf': {'kwargs': {'C': 1.}},
                'data_version_folder': 'dataset_1'
            }
        },

    ]
    model_run_reports = []

    # Add (derived) model trend reports
    label_mapping_values = dict(gender=[0, 1], occupation=[0, 1])
    feature_combinations = pd.DataFrame(
        make_feature_combination_array(label_mapping_values),
        columns=list(label_mapping_values.keys())
    )
    for a_non_trend_artifacts in non_trend_run_artifacts:
        a_run_reports = a_non_trend_artifacts.copy()
        feature_combination_scores = binary_logistic_regression_scores_via_params(
            b=np.array(a_non_trend_artifacts['coefs.json']['b']),
            weights=np.array(a_non_trend_artifacts['coefs.json']['w']),
            X=feature_combinations.values
        )
        feature_combination_score_array = make_feature_combination_score_array(
            feature_combinations=feature_combinations,
            scores=pd.Series(feature_combination_scores)
        )
        a_model_trend_report = make_trend_reports(
            contingency_probabilities=feature_combination_score_array,
            population_subgroup='occupation'
        )
        a_run_reports['model_trend_reports.json'] = a_model_trend_report  # type: ignore
        model_run_reports.append(a_run_reports)

    print(model_run_reports)
    grouped_df = make_trend_grouped_df(model_run_reports, algorithm_class_name='LogisticRegression')

    assert len(grouped_df) == 1
    assert 'dataset_1' in grouped_df.groups

    df = grouped_df.get_group('dataset_1')
    assert len(df) == 4
    assert set(df['occupation_value']) == {0, 1}
    assert set(df['C']) == {0.1, 1.0}
    # NOTE: I know `set` collapses identical values. The duplicates are part of the TDD-by-example
    # philosophy of writing out expressive if strictly unneeded details to ease cognitive load
    assert set(df['trend']) == {0., 0., 1/(1 + exp(1)) - 1/2., 1/2. - 1/(1 + exp(-1))}
    assert set(df['b']) == {0., 0.}
    assert set(df['w_0']) == {0., 1.}
    assert set(df['w_1']) == {-1., 1.}
