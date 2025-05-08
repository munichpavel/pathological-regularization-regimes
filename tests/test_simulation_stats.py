import pandas as pd
from omegaconf import OmegaConf


from prr.simulation_stats import (
    get_sample_size_from_datadir, validate_sample_size,
    make_experiment_run_record, ExperimentRunRecord,
    make_mc_repeat_analysis_df, ExperimentResult
)

def test_get_sample_size_from_datadir():
    # relative_datadir = 'data/samples-200/simpson/800'
    relative_datadir = 'outputs/data-2024-04-23/15-53-11/samples-2400/simpson/42'
    expected = 2400

    sample_size = get_sample_size_from_datadir(relative_datadir)

    assert sample_size == expected


def test_validate_sample_size(tmp_path):
    relative_datadir = 'outputs/data-2024-04-23/15-53-11/samples-2400/simpson/42'

    # test setup
    datadir = tmp_path / relative_datadir
    datadir.mkdir(parents=True)
    data = pd.DataFrame(dict(
        default=2400 * [0],
        gender=2400 * [1],
        occupation=2400 * [42]
    ))
    data.to_csv(datadir / 'default.csv', index=False)

    expected_size = get_sample_size_from_datadir(relative_datadir)
    res = validate_sample_size(relative_datadir, tmp_path, expected_size)
    expected = True
    assert res == expected


def test_get_experiment_run_record(tmp_path):
    relative_run_dir = '2024-03-22/06-57-54'

    # test setup
    run_idx = 0
    run_dir = tmp_path / relative_run_dir
    hydra_run_dir = run_dir / str(run_idx) / '.hydra'
    hydra_run_dir.mkdir(parents=True)

    conf_str = """data_version_folder: outputs/data-2024-04-23/15-53-11/samples-2400/simpson/42
data_batch_idx: 0
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
        batch_idx=0,
        run=relative_run_dir,
    )
    res = make_experiment_run_record(relative_run_dir, run_base=tmp_path, run_idx=0)
    assert res == expected


def test_make_mc_repeat_analysis_df():
    experiment_result_dicts = [
        # size 200
        {'sample_size': 200,
        'fit_intercept': True,
        'batch_idx': 0,
        'adversarial_simpson': 10,
        'never_adversarial_simpson': 0,
        'adversarial_non_simpson': 3,
        'never_adversarial_non_simpson': 7},
        {'sample_size': 200,
        'fit_intercept': True,
        'batch_idx': 1,
        'adversarial_simpson': 10,
        'never_adversarial_simpson': 0,
        'adversarial_non_simpson': 3,
        'never_adversarial_non_simpson': 7},
        {'sample_size': 200,
        'fit_intercept': False,
        'batch_idx': 0,
        'adversarial_simpson': 5,
        'never_adversarial_simpson': 5,
        'adversarial_non_simpson': 7,
        'never_adversarial_non_simpson': 3},
        {'sample_size': 200,
        'fit_intercept': False,
        'batch_idx': 1,
        'adversarial_simpson': 5,
        'never_adversarial_simpson': 5,
        'adversarial_non_simpson': 7,
        'never_adversarial_non_simpson': 3},
        # size 600
         {'sample_size': 600,
        'fit_intercept': True,
        'batch_idx': 0,
        'adversarial_simpson': 10,
        'never_adversarial_simpson': 0,
        'adversarial_non_simpson': 3,
        'never_adversarial_non_simpson': 7},
        {'sample_size': 600,
        'fit_intercept': True,
        'batch_idx': 1,
        'adversarial_simpson': 10,
        'never_adversarial_simpson': 0,
        'adversarial_non_simpson': 3,
        'never_adversarial_non_simpson': 7},
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
