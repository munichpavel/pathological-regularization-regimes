import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union
import datetime
import os
import itertools

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from omegaconf import OmegaConf
import numpy as np


from prr.utils import (
    make_logistic_regression_feature_combination_score_array,
    MODEL_PARAMS_FILENAME,
    TARGET_VALUES,
    is_monotonically_non_decreasing
)
from prr.trends import make_trend_reports


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

OCCUPATION_VALUES = [0, 1]

@dataclass
class ExperimentRunRecord:
    sample_size: int
    fit_intercept: bool
    run: str


@dataclass
class ModelRunSubpopTrendReport:
    fit_intercept: bool
    data_version_folder: str
    sample_size: int
    is_simpson: bool
    C: float
    b: float
    w_0: float
    w_1: float
    subpopulation_name: str
    subpopulation_value: int
    subpopulation_trend: float


@dataclass
class McRun:
    fit_intercept: bool
    sample_size: int
    result_by_dataset: DataFrameGroupBy


@dataclass
class ExperimentResult:
    sample_size: int
    fit_intercept: bool
    adversarial_simpson: int
    never_adversarial_simpson: int
    adversarial_non_simpson: int
    never_adversarial_non_simpson: int


def get_mc_relative_run_parents(
    parent_dirs: list[str],
    modify_time_after: Union[None, datetime.datetime] = None,
    modify_time_before: Union[None, datetime.datetime] = None
) -> list[str]:
    repo_root = Path(os.environ['REPO_ROOT'])
    res = []
    for a_parent in parent_dirs:
        for a_run_dir in (repo_root / a_parent).iterdir():
            if a_run_dir.is_dir():
                res.append(a_run_dir)

    if modify_time_after:
        res = [
            a_run_dir for a_run_dir in res
            if path_modify_time_comparison(a_run_dir, modify_time_after, 'after')
        ]
    if modify_time_before:
        res = [
            a_run_dir for a_run_dir in res
            if path_modify_time_comparison(a_run_dir, modify_time_before, 'before')
        ]

    # Keep only relative to repo-root
    res = [str(a_run_dir.relative_to(repo_root)) for a_run_dir in res]
    return res


def path_modify_time_comparison(a_path: Path, timestamp: datetime.datetime, comparison: str):
    modify_time = a_path.stat().st_ctime
    if comparison == 'after':
        return modify_time > timestamp.timestamp()
    elif comparison == 'before':
        return modify_time < timestamp.timestamp()
    else:
        raise ValueError(f'Valid comparisons are "after" or "before". You entered "{comparison}"')


def make_experiment_run_record(
    relative_run_dir: str, project_root: Path
) -> ExperimentRunRecord:

    with open(project_root / relative_run_dir / '.hydra' / 'config.yaml', 'r') as fp:
        run_conf = OmegaConf.load(fp.name)
    relative_datadir = run_conf.data_version_folder
    sample_size = extract_sample_size_from_data_path(relative_datadir)

    res = ExperimentRunRecord(
        sample_size=sample_size,
        fit_intercept=run_conf.clf.kwargs.fit_intercept,
        run=relative_run_dir,
    )
    return res


def extract_sample_size_from_data_path(data_path: str) -> int:
    data_path_components = data_path.split('/')
    sample_size_components = []
    sample_component_stem = 'samples-'
    for a_component in data_path_components:
        if a_component.startswith(sample_component_stem):
            sample_size_components.append(a_component)
    if len(sample_size_components) != 1:
        raise ValueError(f'Data path does not have component starting with {sample_component_stem}. You entered {data_path}')
    sample_size_component = sample_size_components[0]
    sample_size = int(sample_size_component[len(sample_component_stem):])
    return sample_size


def extract_model_run_subpopulation_trend_reports(
    experiment_run_records: list[ExperimentRunRecord],
    population_subgroup: str
) -> list[ModelRunSubpopTrendReport]:
    res = []
    for a_run_record in experiment_run_records:
        run_dir = repo_root / a_run_record.run
        run_params_by_reg = pd.read_csv(run_dir / MODEL_PARAMS_FILENAME)
        run_config_path = run_dir / '.hydra' / 'config.yaml'
        run_config = OmegaConf.load(run_config_path)
        label_mapping_values = run_config.encoding.label_mapping_values
        data_version_sample_size = extract_sample_size_from_data_path(run_config.data_version_folder)
        data_version_is_simpson = extract_simpsonness_from_data_path(run_config.data_version_folder)

        for a_C_run_params in run_params_by_reg.itertuples():
            feaure_combination_score_array = make_logistic_regression_feature_combination_score_array(
                fit_intercept=a_run_record.fit_intercept,
                b=a_C_run_params.b,
                w=[a_C_run_params.w_0, a_C_run_params.w_1],
                label_mapping_values=label_mapping_values,
                target_values=np.array(TARGET_VALUES)
            )
            run_C_trend_reports = make_trend_reports(
                feaure_combination_score_array, population_subgroup=population_subgroup
            )
            for subpopulation_value in label_mapping_values[population_subgroup]:
                subpop_trend_report = [
                    subpop_report for subpop_report in run_C_trend_reports#['gender_occupation']
                    if subpop_report['population_value'] == subpopulation_value
                ][0]

                a_run_subpopulation_trend_report = ModelRunSubpopTrendReport(
                    fit_intercept=a_run_record.fit_intercept,
                    data_version_folder=run_config.data_version_folder,
                    sample_size=data_version_sample_size,
                    is_simpson=data_version_is_simpson,
                    C=a_C_run_params.C,
                    b=a_C_run_params.b,
                    w_0=a_C_run_params.w_0,
                    w_1=a_C_run_params.w_1,
                    subpopulation_name=population_subgroup,
                    subpopulation_value=subpopulation_value,
                    subpopulation_trend=subpop_trend_report['trend']['gender_trend']
                )
                res.append(a_run_subpopulation_trend_report)
    return res


def extract_simpsonness_from_data_path(data_path: str) -> bool:
    data_path_components = data_path.split('/')
    if 'non-simpson' in data_path_components:
        return False
    elif 'simpson' in data_path_components:
        return True
    else:
        raise ValueError(f'data path must have `simpson` or `non-simpson` as component. You entered {data_path}')


def split_out_mc_runs(results: pd.DataFrame) -> list[McRun]:
    filtered_mc_runs = []
    fit_intercept_values = results['fit_intercept'].unique()
    sample_size_values = results['sample_size'].unique()

    mc_cols = [c for c in results.columns if c not in ['fit_intercept', 'sample_size']]
    for fit_intercept, sample_size in itertools.product(fit_intercept_values, sample_size_values):
        mc_df = results.loc[
            (results['fit_intercept'] == fit_intercept) & (results['sample_size'] == sample_size),
            mc_cols
        ]
        mc_by_dataset = mc_df.groupby('data_version_folder')
        mc_run = McRun(fit_intercept=bool(fit_intercept), sample_size=int(sample_size), result_by_dataset=mc_by_dataset)
        filtered_mc_runs.append(mc_run)
    return filtered_mc_runs


def calculate_experiment_result(fit_intercept: bool, sample_size: int, trend_grouped_df: DataFrameGroupBy) -> ExperimentResult:

    adversarial_reports = classify_dataset_runs_adversariality(trend_grouped_df)
    adversariality_counts = {}
    n_datasets = 0
    for adversariality_group_name, adversariality_report in adversarial_reports.items():
        if adversariality_group_name in ExperimentResult.__dataclass_fields__:
            n_datasets_in_category = len(adversariality_report)
            n_datasets += n_datasets_in_category
            adversariality_counts[adversariality_group_name] = n_datasets_in_category
    res_dict = adversariality_counts
    res_dict['fit_intercept'] = fit_intercept
    res_dict['sample_size'] = sample_size
    res = ExperimentResult(**res_dict)
    return res


def classify_dataset_runs_adversariality(mc_grouped_df: DataFrameGroupBy) -> dict:
    """And collect most adversarial Cs for plotting"""
    adversarial_simpsons = []
    never_adversarial_simpsons = []
    adversarial_non_simpsons = []
    never_adversarial_non_simpsons = []
    for relative_path, df_dataset in mc_grouped_df:
        # create adversarial report
        Cs = {}
        trends = {}
        for occupation_value in OCCUPATION_VALUES:
            occupation_mask = df_dataset['subpopulation_value'] == occupation_value
            Cs[str(occupation_value)] = df_dataset.loc[occupation_mask, 'C']
            trends[str(occupation_value)] = df_dataset.loc[occupation_mask, 'subpopulation_trend']
        adversarial_report = make_adversarial_report(Cs, trends)

        # classify dataset / runs combination
        parent_folder_name = Path(relative_path).parent.name

        # Note: with `np.any` we would have a looser definition of adversariality,
        # requiring trend reversal in only one rather than both subpopulations
        subpopulation_is_adversarial = np.all([
            bool(subpop_report['most_adversarial_Cs'])
            for subpop_report in adversarial_report
        ])

        data_vs_adversarial_report = {}
        # data_vs_adversarial_report['fit_intercept'] = fit_intercept
        # data_vs_adversarial_report['sample_size'] = sample_size
        data_vs_adversarial_report['adversarial_report'] = adversarial_report
        data_vs_adversarial_report['data_version_folder'] = relative_path
        if parent_folder_name == 'simpson':
            if subpopulation_is_adversarial:
                adversarial_simpsons.append(data_vs_adversarial_report)
            else:
                never_adversarial_simpsons.append(data_vs_adversarial_report)
        elif parent_folder_name == 'non-simpson':
            if subpopulation_is_adversarial:
                adversarial_non_simpsons.append(data_vs_adversarial_report)
            else:
                never_adversarial_non_simpsons.append(data_vs_adversarial_report)
        else:
            msg = f'Parent folder name {parent_folder_name} cannot be classified as simpson or non-simpson'
            raise ValueError(msg)

    res = {}
    res['adversarial_simpson'] = adversarial_simpsons
    res['never_adversarial_simpson'] = never_adversarial_simpsons
    res['adversarial_non_simpson'] = adversarial_non_simpsons
    res['never_adversarial_non_simpson'] = never_adversarial_non_simpsons

    # Extract most adversarial C-values
    most_adversarial_Cs = {}
    simpson_most_adversarial_Cs = []
    for a_report in adversarial_simpsons:
        sub_pop_reports = a_report['adversarial_report']
        for a_subpop_report in sub_pop_reports:
            subpop_most_adversarial_C = a_subpop_report['most_adversarial_Cs']['C']
            simpson_most_adversarial_Cs.append(subpop_most_adversarial_C)

    most_adversarial_Cs['simpson_most_adversarial_Cs'] = simpson_most_adversarial_Cs

    non_simpson_most_adversarial_Cs = []
    for a_report in adversarial_non_simpsons:
        sub_pop_reports = a_report['adversarial_report']
        for a_subpop_report in sub_pop_reports:
            subpop_most_adversarial_C = a_subpop_report['most_adversarial_Cs']['C']
            non_simpson_most_adversarial_Cs.append(subpop_most_adversarial_C)

    most_adversarial_Cs['non_simpson_most_adversarial_Cs'] = non_simpson_most_adversarial_Cs

    res['most_adversarial_Cs'] = most_adversarial_Cs
    return res


def make_adversarial_report(Cs: dict[str, np.ndarray], trends: dict[str, np.ndarray]) -> list[dict]:
    report = []
    for occupation_value in OCCUPATION_VALUES:
        occupation_report = {}
        occupation_report['occupation_value'] = occupation_value
        occupation_report['most_adversarial_Cs'] = find_most_adversarial_c(
            Cs=Cs[str(occupation_value)],
            trends=trends[str(occupation_value)]
        )
        # occupation_report['is_adversarial'] = bool(occupation_report['most_adversarial_cs'])
        report.append(occupation_report)

    return report


def find_most_adversarial_c(Cs: np.ndarray, trends: np.ndarray) -> dict:
    if not is_monotonically_non_decreasing(Cs):
        raise ValueError(f'Cs are not monotonically non-decreasing: {Cs}.')
    df = pd.DataFrame(dict(C=Cs, trend=trends))
    trend_largest_C = df.trend.iloc[-1]

    try:
        if trend_largest_C > 0:
            res = df.loc[df.trend < 0].sort_values('trend', ascending=True).iloc[0]
        else:
            res = df.loc[df.trend > 0].sort_values('trend', ascending=False).iloc[0]
        res = dict(res)
    except IndexError as _:
        res = dict()

    return res


def make_mc_repeat_analysis_df(
    experiment_results: list[ExperimentResult]
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
    print(res)
    return res



if __name__ == '__main__':
    """
    Script:

    1. get_mc_relative_run_parents(): Collects all run directories within time window
    2. For each run directory:
    - make_experiment_run_record(): Extracts sample size and fit_intercept settings
    3. extract_model_run_subpopulation_trend_reports(): For each run:
        - Loads model parameters and configs
        - Computes logistic regression trends
    4. split_out_mc_runs(): Groups results by fit_intercept and sample_size
    5. For each group:
        - calculate_experiment_result(): Classifies datasets as adversarial/non-adversarial
    6. make_mc_repeat_analysis_df(): Computes summary statistics across Monte Carlo runs

    Example usage:
    poetry run python -m prr.simulation_stats \
        --rel_mc_run_parents outputs/model-fit-2025-05-27 outputs/model-fit-2025-05-28 \
        --filter_runs_after 2025-05-27T00:00:00 \
        --filter_runs_before 2025-05-28T06:00:00
    """
    import argparse
    from zoneinfo import ZoneInfo
    import json

    import numpy as np


    berlin_tz = ZoneInfo('Europe/Berlin')
    start_time = datetime.datetime.now(berlin_tz)

    parser = argparse.ArgumentParser(description='Multirun script for model-fitting')
    parser.add_argument('--rel_mc_run_parents', nargs='+', help='Relative parent directories holding all mc runs')
    parser.add_argument('--filter_runs_after', type=str, help='ISO timestamp string to filter for only runs after')
    parser.add_argument('--filter_runs_before', type=str, help='ISO timestamp string to filter for only runs before')

    args = parser.parse_args()

    after_time = None
    if args.filter_runs_after:
        after_time = datetime.datetime.fromisoformat(args.filter_runs_after)

    before_time = None
    if args.filter_runs_before:
        before_time = datetime.datetime.fromisoformat(args.filter_runs_before)

    mc_relative_dirs = get_mc_relative_run_parents(
        parent_dirs=args.rel_mc_run_parents,
        modify_time_after=after_time, modify_time_before=before_time
    )

    repo_root = Path(os.environ['REPO_ROOT'])
    now_string = datetime.datetime.now(tz=ZoneInfo('Europe/Berlin')).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = repo_root / 'run-stats' / now_string
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'README.txt', 'w') as fp:
        json.dump(vars(args), fp=fp, indent=2)

    logger.info('Making experiment run records')
    print('Making experiment run records')

    EXPERIMENT_RUN_RECORDS = []
    for relative_run_dir in mc_relative_dirs:
        experiment_run_record = make_experiment_run_record(relative_run_dir, project_root=repo_root)
        EXPERIMENT_RUN_RECORDS.append(experiment_run_record)

    experiment_metadata_df = pd.DataFrame([asdict(experiment_record) for experiment_record in EXPERIMENT_RUN_RECORDS])
    experiment_metadata_df.to_csv(out_dir / 'experiment-metadata.csv', index=False)

    logger.info('Extracting model run subpopulation trend reports')
    print('Extracting model run subpopulation trend reports')
    run_reports = extract_model_run_subpopulation_trend_reports(
        experiment_run_records=EXPERIMENT_RUN_RECORDS,
        population_subgroup='occupation'
    )

    run_report_path = out_dir / 'run-trend-reports.csv'
    run_reports_df = pd.DataFrame([
        asdict(a_run_report) for a_run_report in run_reports
    ])
    run_reports_df.to_csv(run_report_path, index=False)

    logger.info('Calculating mc run analyses')
    print('Calculating mc run analyses')
    split_mc_runs = split_out_mc_runs(run_reports_df)
    mc_experiment_results = []
    for a_mc_run in split_mc_runs:
        mc_experiment_result = calculate_experiment_result(
            a_mc_run.fit_intercept, a_mc_run.sample_size, a_mc_run.result_by_dataset
        )
        mc_experiment_results.append(mc_experiment_result)

    mc_repeat_analysis = make_mc_repeat_analysis_df(mc_experiment_results)
    mc_repeat_analysis.to_csv(out_dir / 'mc-repeat-analysis.csv', index=False)


    end_time = datetime.datetime.now(berlin_tz)
    execution_time = (end_time - start_time).total_seconds()

    log_entry = f"Start Time: {start_time}, End Time: {end_time}, Execution Time: {execution_time} seconds\n"
    logger.info(log_entry)
