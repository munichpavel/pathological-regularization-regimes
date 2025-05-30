import datetime

import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import pytest


from prr.simulation_stats import (
    ExperimentRunRecord,
    ExperimentResult,
    get_mc_relative_run_parents,
    path_modify_time_comparison,
    extract_simpsonness_from_data_path,
    extract_sample_size_from_data_path,
    make_experiment_run_record,
    extract_model_run_subpopulation_trend_reports,
    split_out_mc_runs,
    make_mc_repeat_analysis_df,
    find_most_adversarial_c,

)


def test_path_modify_time_comparision(tmp_path):
    tmp_dir = tmp_path / 'test'
    tmp_dir.mkdir()

    pre_modify_timestamp = datetime.datetime.now()

    a_path = tmp_dir / 'test.txt'
    with open(tmp_dir / a_path, 'w') as fp:
        fp.write('Bagel and lachs.')

    post_modify_timestamp = datetime.datetime.now()

    assert path_modify_time_comparison(a_path, pre_modify_timestamp, 'after')
    assert not path_modify_time_comparison(a_path, pre_modify_timestamp, 'before')

    assert not path_modify_time_comparison(a_path, post_modify_timestamp, 'after')
    assert path_modify_time_comparison(a_path, post_modify_timestamp, 'before')


@pytest.mark.parametrize(
    'data_path,expected,ExpectedException',
    [
        ('some-path/non-simpson/2', False, None),
        ('some-path/data-2024-02-03/simpson/3', True, None),
        ('some-path/some-data/not-really-simpson/4', None, ValueError)
    ]
)
def test_extract_simpsonness_from_data_path(data_path, expected, ExpectedException):
    if ExpectedException:
        with pytest.raises(ExpectedException):
            extract_simpsonness_from_data_path(data_path)
    else:
        res = extract_simpsonness_from_data_path(data_path)
        assert res == expected


@pytest.mark.parametrize(
    'data_path,expected,ExpectedException',
    [
        ('outputs/data-2025-05-06/15-30-35/samples-600/non-simpson/2', 600, None),
        ('outputs/data-2025-05-06/15-30-35/samples-123/simpson/2', 123, None),
        ('some_folder/600/more', None, ValueError),
        ('some_folder/samples_600/more', None, ValueError),
        ('some_folder/600/more', None, ValueError),
        ('some_folder/sample-600/more', None, ValueError),
    ]
)
def test_extract_sample_size_from_data_path(data_path, expected, ExpectedException):
    if ExpectedException:
        with pytest.raises(ExpectedException):
            extract_sample_size_from_data_path(data_path)
    else:
        res = extract_sample_size_from_data_path(data_path)
        assert res == expected

def test_get_experiment_run_record(tmp_path):
    relative_run_dir = f'2024-03-22/06-57-54/'

    # test setup

    run_dir = tmp_path / relative_run_dir
    hydra_run_dir = run_dir / '.hydra'
    hydra_run_dir.mkdir(parents=True)

    conf_str = """data_version_folder: outputs/data-2024-04-23/15-53-11/samples-2400/simpson/42
clf:
  parent_module: sklearn.linear_model
  class_name: LogisticRegression
  kwargs:
    C: 1.0e-07
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
    - 1
"""
    run_conf = OmegaConf.create(conf_str)
    with open(hydra_run_dir / 'config.yaml', 'w') as fp:
        OmegaConf.save(run_conf, f=fp.name)

    # the test
    expected = ExperimentRunRecord(
        sample_size=2400,
        fit_intercept=True,
        # batch_idx=0,
        run=relative_run_dir,
    )
    res = make_experiment_run_record(relative_run_dir, project_root=tmp_path)
    assert res == expected


def test_find_most_adversarial_c():
    cs = np.array([0.1, 1., 10.])
    trends = np.array([-1., 0., 0.5])
    expected = dict(C=0.1, trend=-1)

    res = find_most_adversarial_c(cs, trends)
    assert res == expected


def test_make_mc_repeat_analysis_df():
    experiment_result_dicts = [
        # size 200
        {
            'sample_size': 200,
            'fit_intercept': True,
            'adversarial_simpson': 10,
            'never_adversarial_simpson': 0,
            'adversarial_non_simpson': 3,
            'never_adversarial_non_simpson': 7
        },
        {
            'sample_size': 200,
            'fit_intercept': True,
            'adversarial_simpson': 10,
            'never_adversarial_simpson': 0,
            'adversarial_non_simpson': 3,
            'never_adversarial_non_simpson': 7
        },
        {
            'sample_size': 200,
            'fit_intercept': False,
            'adversarial_simpson': 5,
            'never_adversarial_simpson': 5,
            'adversarial_non_simpson': 7,
            'never_adversarial_non_simpson': 3
        },
        {
            'sample_size': 200,
            'fit_intercept': False,
            'adversarial_simpson': 5,
            'never_adversarial_simpson': 5,
            'adversarial_non_simpson': 7,
            'never_adversarial_non_simpson': 3
        },
        # size 600
        {
            'sample_size': 600,
            'fit_intercept': True,
            'adversarial_simpson': 10,
            'never_adversarial_simpson': 0,
            'adversarial_non_simpson': 3,
            'never_adversarial_non_simpson': 7
        },
        {
            'sample_size': 600,
            'fit_intercept': True,
            'adversarial_simpson': 10,
            'never_adversarial_simpson': 0,
            'adversarial_non_simpson': 3,
            'never_adversarial_non_simpson': 7
        },
    ]
    experiment_results = [
        ExperimentResult(**a_record) for a_record in experiment_result_dicts
    ]

    expected = pd.DataFrame(
        [

            dict(
                sample_size=200,
                fit_intercept=False,
                mean_ratio_adversarial_simpson=0.5,
                std_ratio_adversarial_simpson=0.,
                mean_ratio_adversarial_non_simpson=0.7,
                std_ratio_adversarial_non_simpson=0.
            ),
            dict(
                sample_size=200,
                fit_intercept=True,
                mean_ratio_adversarial_simpson=1.,
                std_ratio_adversarial_simpson=0.,
                mean_ratio_adversarial_non_simpson=0.3,
                std_ratio_adversarial_non_simpson=0.
            ),
             dict(
                sample_size=600,
                fit_intercept=True,
                mean_ratio_adversarial_simpson=1.,
                std_ratio_adversarial_simpson=0.,
                mean_ratio_adversarial_non_simpson=0.3,
                std_ratio_adversarial_non_simpson=0.
            ),
        ]
    )
    df = make_mc_repeat_analysis_df(experiment_results)

    pd.testing.assert_frame_equal(df, expected)


def test_get_mc_relative_dirs(tmp_path, monkeypatch):
    # Simulation folder structure setup
    base_dir = tmp_path / 'root'
    base_dir.mkdir()
    monkeypatch.setenv('REPO_ROOT', base_dir.as_posix())

    run_day_dirs = ['output/model-fit-day-1', 'output/model-fit-day-2']
    run_time_subdirs = ['hh-mm-00', 'hh-mm-10']
    # n_runs_per_batch = 1
    modify_time_after = datetime.datetime.now()

    for a_run_day_dir in run_day_dirs:
        for a_rel_run_dir in run_time_subdirs:
            a_run_dir = base_dir / a_run_day_dir / a_rel_run_dir
            a_run_dir.mkdir(parents=True)
            # for idx in range(n_runs_per_batch):
            #     child_run_dir = base_dir / a_run_day_dir / a_run_dir / str(idx)
            #     child_run_dir.mkdir(parents=True)

    modify_time_before = datetime.datetime.now()

    later_run_time_subdir = 'hh-mm-59'
    excluded_subdir = base_dir / run_day_dirs[-1] / later_run_time_subdir
    excluded_subdir.mkdir(parents=True)

    mc_relative_run_parents = get_mc_relative_run_parents(
        parent_dirs=run_day_dirs,
        modify_time_after=modify_time_after, modify_time_before=modify_time_before
    )

    assert len(mc_relative_run_parents) == len(run_day_dirs) * len(run_time_subdirs)# * n_runs_per_batch


