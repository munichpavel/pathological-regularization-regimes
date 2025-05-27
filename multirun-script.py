import concurrent.futures
from typing import List, Tuple
import logging
import os
from pathlib import Path
import json
import subprocess

from omegaconf import OmegaConf


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_batch(
    batch_data: Tuple[List[str], List[str], int, bool, Path]
) -> None:
    simpson_dirs, non_simpson_dirs, batch_idx, fit_intercept, repo_root = batch_data
    rel_data_dirs = simpson_dirs + non_simpson_dirs
    rel_data_dir_strs = ','.join(rel_data_dirs)

    logger.info(
        f'Batch {batch_idx}: Processing {len(simpson_dirs)} simpson and '
        f'{len(non_simpson_dirs)} non-simpson datasets'
    )

    subprocess.run([
        'poetry', 'run', 'python', 'main.py',
        '--multirun',
        f'data_version_folder={rel_data_dir_strs}',
        f'data_batch_idx={batch_idx}',
        f'clf.kwargs.fit_intercept={str(fit_intercept).lower()}',
        '+hydra.job.chdir=True',
        '+hydra.sweeper.max_batch_size=10',
    ])


def main(
    rel_data_parent: str,
    max_workers: int = None  # None means use all available cores
) -> None:
    # Sweep over data samples
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

        with open('conf/base-model-fitting.yaml', 'r') as fp:
            base_config = OmegaConf.load(fp.name)

        logger.info(
            'Starting multirun with base config '
            f'{json.dumps(OmegaConf.to_container(base_config), indent=2)}'
        )

        batch_size = base_config.n_datasets_per_batch
        n_batches = data_gen_config.n_datasets / batch_size
        if n_batches.is_integer():
            n_batches = int(n_batches)
        else:
            raise ValueError(
                f'Batch size {batch_size} does not divide '
                f'number-of-datasets {data_gen_config.n_datasets}'
            )

        batch_tasks = []
        for data_batch_idx in range(n_batches):
            batch_simpson_dirs = rel_simpson_data_dirs[
                data_batch_idx * batch_size: (data_batch_idx + 1) * batch_size
            ]
            batch_non_simpson_dirs = rel_non_simpson_data_dirs[
                data_batch_idx * batch_size: (data_batch_idx + 1) * batch_size
            ]

            for fit_intercept in [True, False]:
                batch_tasks.append((
                    batch_simpson_dirs,
                    batch_non_simpson_dirs,
                    data_batch_idx,
                    fit_intercept,
                    repo_root
                ))

        # Process batches in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(process_batch, batch_tasks))

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
