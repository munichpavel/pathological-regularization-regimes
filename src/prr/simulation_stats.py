from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from omegaconf import OmegaConf

from prr.docs import get_multirun_reports, make_trend_grouped_df, classify_dataset_runs_adversariality


@dataclass
class ExperimentRunRecord:
    sample_size: int
    fit_intercept: bool
    batch_idx: int
    run: str


@dataclass
class ExperimentResult:
    sample_size: int
    fit_intercept: bool
    batch_idx: int
    adversarial_simpson: int
    never_adversarial_simpson: int
    adversarial_non_simpson: int
    never_adversarial_non_simpson: int


def get_sample_size_from_datadir(relative_datadir: str) -> int:
    datadir_components = relative_datadir.split('/')
    sample_size_component = datadir_components[3]
    last_hyphen_component = sample_size_component.split('-')[-1]
    res = int(last_hyphen_component)
    return res


def validate_sample_size(relative_datadir: str, data_root: Path, expected_size: int) -> bool:
    df = pd.read_csv(data_root / relative_datadir / 'default.csv')
    res = df.shape[0] == expected_size
    return res


def make_experiment_run_record(
    relative_run_dir: str, run_base: Path, run_idx = 0
) -> ExperimentRunRecord:

    with open(run_base / relative_run_dir / str(run_idx) / '.hydra' / 'config.yaml', 'r') as fp:
        run_conf = OmegaConf.load(fp.name)
    relative_datadir = run_conf.data_version_folder
    sample_size = get_sample_size_from_datadir(relative_datadir)

    res = ExperimentRunRecord(
        sample_size=sample_size,
        fit_intercept=run_conf.clf.kwargs.fit_intercept,
        batch_idx=run_conf.data_batch_idx,
        run=relative_run_dir,
    )
    return res


def calculate_experiment_result(run: str, run_root: Path) -> Dict:
    run_dir = run_root / run

    trend_reports = get_multirun_reports(multirun_dir=run_dir)
    trend_df = make_trend_grouped_df(trend_reports, algorithm_class_name='LogisticRegression')

    adversarial_results = classify_dataset_runs_adversariality(trend_df)
    adversariality_counts = {}
    n_datasets = 0
    for adversariality_group_name, adversariality_results in adversarial_results.items():
        n_datasets_in_category = len(adversariality_results)
        n_datasets += n_datasets_in_category
        adversariality_counts[adversariality_group_name] = n_datasets_in_category

    return adversariality_counts


def make_mc_repeat_analysis_df(
    experiment_results: List[ExperimentResult]
) -> pd.DataFrame:

    results_df = pd.DataFrame(asdict(a_result) for a_result in experiment_results)
    n_simpson = results_df.adversarial_simpson + results_df.never_adversarial_simpson
    results_df['ratio_adversarial_simpson'] = results_df.adversarial_simpson / n_simpson

    n_non_simpson = results_df.adversarial_non_simpson + results_df.never_adversarial_non_simpson
    results_df['ratio_adversarial_non_simpson'] = results_df.adversarial_non_simpson / n_non_simpson

    # mc_cols = ['sample_size', 'fit_intercept', 'batch_idx', 'ratio_adversarial_simpson', 'ratio_adversarial_non_simpson']
    value_cols = ['ratio_adversarial_simpson', 'ratio_adversarial_non_simpson']
    groupby_cols = ['sample_size', 'fit_intercept']
    mc_dfs = []
    for value_col in value_cols:
        mc_mean = results_df.groupby(groupby_cols)[value_col].mean()
        mc_std = results_df.groupby(groupby_cols)[value_col].std()


        # results_df.loc[results_df.fit_intercept == True]
        mc_mean.name = 'mean_' + mc_mean.name
        mc_mean_df = mc_mean.to_frame()
        mc_dfs.append(mc_mean_df)

        mc_std.name = 'std_' + mc_std.name
        mc_std_df = mc_std.to_frame()
        mc_dfs.append(mc_std_df)

    res = mc_dfs[0]
    for df_idx in range(1, len(mc_dfs)):
        res = pd.merge(left=res, right=mc_dfs[df_idx], left_index=True, right_index=True)
    res = res.reset_index()
    return res
