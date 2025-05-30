import concurrent.futures
from typing import List, Tuple, Union
import logging
import os
from pathlib import Path
import json
import subprocess
import itertools
from datetime import datetime

from omegaconf import OmegaConf


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# def process_batch(
#     batch_data: Tuple[List[str], List[str], int]
# ) -> None:
#     simpson_dirs, non_simpson_dirs, a_batch_idx = batch_data
#     rel_data_dirs = simpson_dirs + non_simpson_dirs
#     rel_data_dir_strs = ','.join(rel_data_dirs)

#     logger.info(
#         f'Batch {a_batch_idx}: Processing {len(simpson_dirs)} simpson and '
#         f'{len(non_simpson_dirs)} non-simpson datasets'
#     )
#     subprocess.run([
#         'poetry', 'run', 'python', 'main.py',
#         '--multirun',
#         f'data_version_folder={rel_data_dir_strs}',
#         f'data_batch_idx={a_batch_idx}',
#         f'clf.kwargs.fit_intercept=false,true',
#         '+hydra.job.chdir=True',
#     ])

def process_dataset(
    # rel_data_dir: str
    batch_data: Tuple[str, str]
) -> None:
    rel_data_dir, fit_intercept = batch_data

    # Create unique run directory based on timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f")
    run_dir = f"outputs/model-fit-{timestamp}"

    logger.info(f'Processing dataset {rel_data_dir} with fit_intercept={fit_intercept} in {run_dir}')

    subprocess.run([
        'poetry', 'run', 'python', 'main.py',
        f'data_version_folder={rel_data_dir}',
        f'clf.kwargs.fit_intercept={fit_intercept}',
        '+hydra.job.chdir=True',
        f'++hydra.run.dir={run_dir}'
    ])

def main(
    rel_data_parent: str,
    max_workers: Union[None, int] = None  # None means use all available cores
) -> None:
    repo_root = Path(os.environ['REPO_ROOT'])
    data_parent = repo_root / rel_data_parent
    folder_name = 'samples-{sample_size}'

    # open data-generation config
    with open(Path(rel_data_parent) / '.hydra/config.yaml', 'r') as fp:
        data_gen_config = OmegaConf.load(fp.name)

    logger.info(
        'Using generated data with config '
        f'{json.dumps(OmegaConf.to_container(data_gen_config), indent=2)}'
    )

    for sample_size in data_gen_config.sample_sizes:
        data_root = data_parent / folder_name.format(
            sample_size=sample_size
        )

        simpson_data_root = data_root / 'simpson'
        non_simpson_data_root = data_root / 'non-simpson'

        rel_simpson_data_dirs = [
            datadir.relative_to(repo_root).as_posix()
            for datadir in simpson_data_root.iterdir()
        ]
        rel_non_simpson_data_dirs = [
            datadir.relative_to(repo_root).as_posix()
            for datadir in non_simpson_data_root.iterdir()
        ]
        all_rel_data_dirs = rel_simpson_data_dirs + rel_non_simpson_data_dirs

        with open('conf/base-model-fitting.yaml', 'r') as fp:
            base_config = OmegaConf.load(fp.name)

        logger.info(
            'Starting runs with base config '
            f'{json.dumps(OmegaConf.to_container(base_config), indent=2)}'
        )

        fit_intercept_vals = ['true', 'false']
        process_dataset_args = list(itertools.product(all_rel_data_dirs, fit_intercept_vals))
        logger.info(f'Running dataset processing for {len(process_dataset_args)} combinations')
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(process_dataset, process_dataset_args))

if __name__ == '__main__':
    """
    Usage examples:

    poetry run python multirun-script.py \
        --rel_data_parent outputs/data-2024-04-23/15-53-11

    """
    import argparse

    parser = argparse.ArgumentParser(description='Multirun script for model-fitting')
    parser.add_argument('--rel_data_parent', type=str, help='Data parent folder relative to repo root')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of parallel workers')
    args = parser.parse_args()
    main(args.rel_data_parent, args.max_workers)
