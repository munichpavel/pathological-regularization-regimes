from pathlib import Path
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Union

import logging
import json

from omegaconf import OmegaConf, ListConfig, DictConfig
import yaml

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
import numpy as np

import matplotlib.pyplot as plt

from .utils import generate_log_spaced_array, is_monotonically_non_decreasing


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OCCUPATION_VALUES = [0, 1]  # TODO extract from base-model config

INTRO = '''# Adversarial Simpson Discussion Document
'''

@dataclass
class Example:
    rel_path: str
    notes: str


def clear_discussion_dir(discussion_dir: Path) -> None:
        if discussion_dir.exists():
            shutil.rmtree(discussion_dir)
        discussion_dir.mkdir(parents=True)
        (discussion_dir / 'graphics').mkdir()


def extract_discussion_config_params(a_config: Union[DictConfig, ListConfig]) -> Dict:
    discussion_params = {}
    discussion_params['algorithm_class_name'] = a_config.clf.class_name
    discussion_params['fit_intercept'] = a_config.clf.kwargs.fit_intercept
    discussion_params['solver'] = a_config.clf.kwargs.solver

    return discussion_params


def make_intro(relative_run_dir: str, zeroeth_config: dict) -> str:
    intro_text = INTRO
    intro_text += (
        f'\n\nRelative run directory: {relative_run_dir}'
        f'\n\nShared run config params:\n\n{zeroeth_config}'
        '\n\n## Examples'
        '\n\n'
    )

    return intro_text


def write_intro(doc_path: Path, intro_text: str) -> None:
    with open(doc_path, 'w') as fp:
        fp.write(intro_text)


def write_examples(grouped_df: DataFrameGroupBy, doc_path: Path, example_data: List[Example]) -> None:

    for an_example in example_data:
        example_values = grouped_df.get_group(an_example.rel_path)
        Cs = {}
        trends = {}
        for occupation_value in OCCUPATION_VALUES:
            occupation_mask = example_values['occupation_value'] == occupation_value
            Cs[str(occupation_value)] = example_values.loc[occupation_mask, 'C']
            trends[str(occupation_value)] = example_values.loc[occupation_mask, 'trend']
        out_filename = convert_rel_path_to_filename(an_example.rel_path)
        out_dir = doc_path.parent
        out_path = out_dir / 'graphics' / out_filename
        plot_trend_vs_c(Cs, trends, out_path)

        # Note: the below are independent of occupation value,
        # so we just take the last
        params_df = example_values.loc[occupation_mask, ['b', 'w_0', 'w_1']]  # type: ignore
        params = {}
        for param_name in ['b', 'w_0', 'w_1']:
            params[param_name] = params_df[param_name].values
        plot_fitted_params_vs_c(
            Cs[str(occupation_value)],  # type: ignore
            params,
            out_path = out_dir / 'graphics' / ('params-' + out_filename)
        )

        adversarial_report = make_adversarial_report(Cs, trends)

        example_section = make_example_section(an_example, adversarial_report, out_dir)

        with open(doc_path, 'a') as fp:
            fp.write(example_section)


def get_multirun_reports(multirun_dir: Path) -> List:

    artifact_filenames = [
        'model_trend_reports.json',
        'coefs.json',
        '.hydra/config.yaml'
    ]

    model_trend_reports = get_multirun_artifacts(multirun_dir, artifact_filenames)
    return model_trend_reports


def get_multirun_artifacts(
    multirun_dir: Path, artifact_filenames: List[str]
) -> List[Dict]:
    multirun_artifacts = []

    for p in multirun_dir.iterdir():
        try:
            if p.is_dir():
                run_artifacts = {}

                for filename in artifact_filenames:
                    with open(p / filename, 'r') as fp:
                        suffix = Path(filename).suffix
                        if suffix == '.json':
                            artifact = json.load(fp)
                        elif suffix in ['.yaml', '.yml']:
                            artifact = yaml.safe_load(fp)
                        else:
                            msg = f'Opening artifact with suffix {suffix} not implemented'
                            raise NotImplementedError(msg)
                    run_artifacts[filename] = artifact
                multirun_artifacts.append(run_artifacts)

        except FileNotFoundError as err:
            logger.warning(err)


    return multirun_artifacts


# def make_trend_grouped_df(model_trend_reports: List, algorithm_class_name: str) -> DataFrameGroupBy:
#     regularization_vs_trend_per_dataset = []
#     for occupation_value in OCCUPATION_VALUES:
#         for run_reports in model_trend_reports:
#             reg_vs_trend = {}
#             trend_report = run_reports['model_trend_reports.json']
#             subpop_trend_report = [
#                 subpop_report for subpop_report in trend_report#['gender_occupation']
#                 if subpop_report['population_value'] == occupation_value
#             ][0]
#             reg_vs_trend['occupation_value'] = occupation_value
#             trend_subpop = subpop_trend_report['trend']['gender_trend']
#             reg_vs_trend['trend'] = trend_subpop

#             # Note: the below values are independent of occupation value, so values
#             # will be duplicated across rows
#             run_config = run_reports['.hydra/config.yaml']
#             coef_report = run_reports['coefs.json']

#             if algorithm_class_name == 'LogisticRegression':
#                 reg_vs_trend['b'] = coef_report['b'][0]
#                 reg_vs_trend['w_0'] = coef_report['w'][0][0]
#                 reg_vs_trend['w_1'] = coef_report['w'][0][1]

#                 C = run_config['clf']['kwargs']['C']
#             else:
#                 raise NotImplementedError(f'for {algorithm_class_name}')

#             reg_vs_trend['C'] = C
#             reg_vs_trend['data_version_folder'] = run_config['data_version_folder']

#             regularization_vs_trend_per_dataset.append(reg_vs_trend)
#     # regularization_vs_trend_per_dataset
#     df = pd.DataFrame(regularization_vs_trend_per_dataset)
#     grouped = df.sort_values('C').groupby('data_version_folder')

#     return grouped


def convert_rel_path_to_filename(rel_path: str, extension='.png') -> str:
    res = rel_path.replace('/', '-') + extension
    return res


PLT_MARKER_SIZE = 4
PLT_MARKER = '.'


def plot_trend_vs_c(Cs: Dict[str, np.ndarray], trends: Dict[str, np.ndarray], out_path: Path) -> None:

    plt.rcParams['text.usetex'] = True
    fig, axs = plt.subplots(ncols=1, nrows=2, layout="constrained")
    occupation_value = 0
    # fig.suptitle('Trend vs c values', fontsize=16)

    y_labels = {
        0: r"$\mathcal{T}_{X_2=0}$",
        1: r"$\mathcal{T}_{X_2=1}$"
    }
    for occupation_value in OCCUPATION_VALUES:
        cs  = 1 / Cs[str(occupation_value)]
        axs[occupation_value].plot(
            cs, trends[str(occupation_value)],
            marker=PLT_MARKER, linestyle='-', markersize=PLT_MARKER_SIZE,
            color='#4169E1' #'#6495ED'
        )
        axs[occupation_value].set_xscale('log')
        # axs[occupation_value].set_xlabel(f'c values, occupation {occupation_value}')
        axs[occupation_value].set_xlabel(f'c')
        # axs[occupation_value].set_ylabel('trend')
        axs[occupation_value].set_ylabel(y_labels[occupation_value])
        axs[occupation_value].grid(False)
        axs[occupation_value].plot(cs, np.zeros(len(cs)), color='black', linestyle='dashed')


    plt.savefig(out_path, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_fitted_params_vs_c(Cs: np.ndarray, params: Dict[str, np.ndarray], out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    cs = 1 / Cs
    plt.suptitle('Fitted parameters vs c values', fontsize=16)

    # Plot for 'b'
    plt.plot(
        cs, params['b'],
        marker=PLT_MARKER, linestyle='-', markersize=PLT_MARKER_SIZE,
        color='b', label='b'
    )

    # Plot for 'w_0'
    plt.plot(
        cs, params['w_0'],
        marker=PLT_MARKER, linestyle='-', markersize=PLT_MARKER_SIZE,
        color='r', label='w_0'
    )

    # Plot for 'w_1'
    plt.plot(
        cs, params['w_1'],
        marker=PLT_MARKER, linestyle='-', markersize=PLT_MARKER_SIZE,
        color='g', label='w_1'
    )

    # Log scale for better visualization when C values vary widely
    plt.xscale('log')
    plt.xlabel('c')
    plt.ylabel('Values')
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


# def make_adversarial_report(Cs: Dict[str, np.ndarray], trends: Dict[str, np.ndarray]) -> List[Dict]:
#     report = []
#     for occupation_value in OCCUPATION_VALUES:
#         occupation_report = {}
#         occupation_report['occupation_value'] = occupation_value
#         occupation_report['most_adversarial_Cs'] = find_most_adversarial_c(
#             Cs=Cs[str(occupation_value)],
#             trends=trends[str(occupation_value)]
#         )
#         # occupation_report['is_adversarial'] = bool(occupation_report['most_adversarial_cs'])
#         report.append(occupation_report)

#     return report


# def find_most_adversarial_c(Cs: np.ndarray, trends: np.ndarray) -> Dict:
#     if not is_monotonically_non_decreasing(Cs):
#         raise ValueError(f'Cs are not monotonically non-decreasing: {Cs}.')
#     df = pd.DataFrame(dict(C=Cs, trend=trends))
#     trend_largest_C = df.trend.iloc[-1]

#     try:
#         if trend_largest_C > 0:
#             res = df.loc[df.trend < 0].sort_values('trend', ascending=True).iloc[0]
#         else:
#             res = df.loc[df.trend > 0].sort_values('trend', ascending=False).iloc[0]
#         res = dict(res)
#     except IndexError as _:
#         res = dict()

#     return res


def make_example_section(an_example: Example, adversarial_report: List[Dict], out_dir: Path) -> str:
    graphic_filename = convert_rel_path_to_filename(an_example.rel_path)

    intro = (
        f'\n### {an_example.rel_path}'
        f'\n\n{an_example.notes}'
    )

    adversarial_subsection = '\n\n**Adversarial report**'
    for occupation_report in adversarial_report:
        adversarial_subsection += '\n\n' + str(occupation_report)

    plot_trend = (
        '\n \pagebreak \n'
        f'\n\n ![plot-rr-{an_example.rel_path}]({out_dir.as_posix()}/graphics/{graphic_filename})'
        # '\n \pagebreak \n'
    )
    plot_params = (
        f'\n\n ![plot-params-{an_example.rel_path}]({out_dir.as_posix()}/graphics/{"params-" + graphic_filename})'
        '\n \pagebreak \n'
    )

    section = intro + adversarial_subsection + plot_trend + plot_params

    return section


def write_dataset_runs_adversariality_section(grouped_df: DataFrameGroupBy, doc_path: Path) -> None:
    dataset_runs_adversarility_classification = classify_dataset_runs_adversariality(grouped_df)
    most_adversarial_Cs = dataset_runs_adversarility_classification['most_adversarial_Cs']
    del dataset_runs_adversarility_classification['most_adversarial_Cs']

    section_header = '\n\n## Adversariality classification, all datasets\n\n'

    with open(doc_path, 'a') as fp:
        fp.write(section_header)

    adversariality_counts = {}
    n_datasets = 0
    for adversariality_group_name, adversariality_results in dataset_runs_adversarility_classification.items():
        n_datasets_in_category = len(adversariality_results)
        n_datasets += n_datasets_in_category
        adversariality_counts[adversariality_group_name] = n_datasets_in_category

    with open(doc_path, 'a') as fp:
        fp.write('\n\n#### Overview')
        fp.write(f'\n\n Of {n_datasets} datasets, the breakdown of counts is\n\n')
        json.dump(adversariality_counts, fp, indent=2)
        fp.write('\n \pagebreak \n')

    # Plot histogram of most-adversarial C-values
    out_dir = doc_path.parent
    for C_key in ['simpson_most_adversarial_Cs', 'non_simpson_most_adversarial_Cs']:
        out_path = out_dir / 'graphics' / (C_key + '.png')
        Cs = most_adversarial_Cs[C_key]
        plot_most_adversarial_Cs(Cs, out_path)
        histogram_text = (
            f'\n\n#### {C_key}\n\n ![plot-C-histogram-{C_key}]({out_path})'
            '\n\pagebreak'
        )
        with open(doc_path, 'a') as fp:
            fp.write(histogram_text)

    # Write data-set level results, truncated to a write-limit
    write_limit = 200
    for adversariality_group_name, adversariality_results in dataset_runs_adversarility_classification.items():
        subsection_header = f'\n\n#### {adversariality_group_name}\n\n'

        with open(doc_path, 'a') as fp:
            fp.write(subsection_header)
            json.dump(adversariality_results[:write_limit], fp, indent=2)



    # write to disk for further plots / analyses
    plot_data_dir = out_dir / 'plot-data'
    plot_data_dir.mkdir()
    with open(plot_data_dir / 'most-adversarial-Cs.json', 'w') as fp:
        json.dump( most_adversarial_Cs, fp)


def plot_most_adversarial_Cs(Cs: Union[List[float], np.ndarray], out_path: Path) -> None:
    # Convert Cs to numpy array for easier manipulation
    Cs = np.array(Cs)

    plt.figure(figsize=(10, 6))
    bins = generate_log_spaced_array(-7, 7, 50)  # may need  adjustment
    plt.hist(Cs, bins=bins.tolist(), edgecolor='black', alpha=0.7)

    # Set x-axis to log scale
    plt.xscale('log')

    # Add title and labels
    plt.title('Histogram of Cs', fontsize=15)
    plt.xlabel('C values (log scale)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Save the plot to the specified path
    plt.savefig(out_path)
    plt.close()


def classify_dataset_runs_adversariality(grouped_df: DataFrameGroupBy) -> Dict:
    """And collect most adversarial Cs for plotting"""
    adversarial_simpsons = []
    never_adversarial_simpsons = []
    adversarial_non_simpsons = []
    never_adversarial_non_simpsons = []
    for relative_path, df_dataset in grouped_df:
        # create adversarial report
        Cs = {}
        trends = {}
        for occupation_value in OCCUPATION_VALUES:
            occupation_mask = df_dataset['occupation_value'] == occupation_value
            Cs[str(occupation_value)] = df_dataset.loc[occupation_mask, 'C']
            trends[str(occupation_value)] = df_dataset.loc[occupation_mask, 'trend']
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
    res['adversarial_simpsons'] = adversarial_simpsons
    res['never_adversarial_simpsons'] = never_adversarial_simpsons
    res['adversarial_non_simpsons'] = adversarial_non_simpsons
    res['never_adversarial_non_simpsons'] = never_adversarial_non_simpsons

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


def write_pdf(doc_path: Path, out_path) -> None:
     subprocess.run(
          ['pandoc', doc_path.as_posix(), '-o', out_path.as_posix()]
     )


if __name__ == '__main__':
    """Example usage

    poetry run python -m prr.docs \
        --relative_run_dir multirun/model-fit-2024-06-15/13-47-22
        --relative_example_data_config_path docs/example-data-paper.json
    """
    import argparse
    import os
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description='Compile run documentation and analyses')

    # Add arguments
    parser.add_argument('--relative_run_dir', type=str, help='Relative folder for run')
    parser.add_argument('--relative_example_data_config_path', type=str, help='Relative path to example dataset specification')
    # Parse the arguments
    args = parser.parse_args()

    supported_algorithms = ['LogisticRegression']

    algorithm_run_combinations = {
        'LogisticRegression': [
            args.relative_run_dir
        ]
    }

    REPO_ROOT = Path(os.environ['REPO_ROOT'])
    DISCUSSION_ROOT = REPO_ROOT / 'docs' / 'discussion'
    with open(REPO_ROOT / args.relative_example_data_config_path, 'r') as fp:
        example_data = json.load(fp)
    example_data = [Example(**an_example) for an_example in example_data]

    for algorithm_class_name in algorithm_run_combinations.keys():
        for run in algorithm_run_combinations[algorithm_class_name]:
            logger.info(f'Selected model: {algorithm_class_name}')

            # multirun_dir = REPO_ROOT / 'fit-model' / 'multirun' / run
            multirun_dir = REPO_ROOT / run
            zeroth_run_config_path = multirun_dir / '0' / '.hydra' / 'config.yaml'
            logger.info(f'Loading zeroth run config file {zeroth_run_config_path.as_posix()}')
            with open(zeroth_run_config_path, 'r') as fp:
                zeroeth_config = OmegaConf.load(fp.name)

            classifier_kwargs_keys = zeroeth_config.clf.kwargs.keys()
            if algorithm_class_name == 'LogisticRegression':
                assert 'C' in classifier_kwargs_keys, f'Expected parameter C among kwargs for {algorithm_class_name}; got {classifier_kwargs_keys}'
            else:
                raise NotImplementedError(algorithm_class_name)

            discussion_config_params = extract_discussion_config_params(zeroeth_config)
            logger.info(f'Generating simpson examples doc for {discussion_config_params}')

            intro_text = make_intro(relative_run_dir=run, zeroeth_config=discussion_config_params)

            TEXT_FILENAME_STEM = 'examples'
            discussion_filename = '-'.join([
                TEXT_FILENAME_STEM,
                algorithm_class_name,
                'fitintercept',
                str(discussion_config_params['fit_intercept'])
            ])
            discussion_dir = DISCUSSION_ROOT / discussion_filename

            logger.info('Removing previous artifacts directory')
            clear_discussion_dir(discussion_dir=discussion_dir)
            md_filename = discussion_filename + '.md'
            out_filename = discussion_filename + '.pdf'

            doc_path = discussion_dir / md_filename
            write_intro(
                doc_path=doc_path,
                intro_text=intro_text
            )
            logger.info(f'Making multirun reports from {multirun_dir}')
            model_trend_reports = get_multirun_reports(multirun_dir=multirun_dir)

            logger.info(f'Putting reports in a dataframe')
            grouped_df = make_trend_grouped_df(model_trend_reports, algorithm_class_name)

            logger.info('Writing examples')
            write_examples(
                grouped_df=grouped_df,
                doc_path=doc_path,
                example_data=example_data
            )

            logger.info('Writing dataset vs runs adversariality overview')
            write_dataset_runs_adversariality_section(grouped_df=grouped_df, doc_path=doc_path)

            logger.info(f'Writing pdf to {discussion_dir / out_filename}')
            write_pdf(
                doc_path=discussion_dir / md_filename,
                out_path=discussion_dir / out_filename
            )
