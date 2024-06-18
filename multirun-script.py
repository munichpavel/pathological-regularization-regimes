import logging
import os
from pathlib import Path
import json
import subprocess

from omegaconf import OmegaConf

import numpy as np

from arr.utils import generate_log_spaced_array
from arr.docs import is_monotonically_increasing

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(
    C_start_power_reg_0: int, C_stop_power_reg_0: int, n_Cs_reg_0: int,
    C_start_power_reg_1: int, C_stop_power_reg_1: int, n_Cs_reg_1: int,
    C_start_power_reg_2: int, C_stop_power_reg_2: int, n_Cs_reg_2: int,
    rel_data_parent: str
) -> None:
    Cs_reg_0 = generate_log_spaced_array(C_start_power_reg_0, C_stop_power_reg_0, n_Cs_reg_0)
    Cs_reg_1 = generate_log_spaced_array(C_start_power_reg_1, C_stop_power_reg_1, n_Cs_reg_1)
    Cs_reg_2 = generate_log_spaced_array(C_start_power_reg_2, C_stop_power_reg_2, n_Cs_reg_2)
    Cs = np.concatenate([Cs_reg_0, Cs_reg_1, Cs_reg_2])
    if not is_monotonically_increasing(Cs):
        print(Cs)
        raise ValueError("Regularization parameters must be monotocically increasing")
    C_strings = ','.join([str(C) for C in Cs])

    regularization_overwrite = f'clf.kwargs.C={C_strings}'

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
        n_batches =  data_gen_config.n_datasets / batch_size
        if n_batches.is_integer():
            n_batches = int(n_batches)
        else:
            raise ValueError(
                f'Batch size {batch_size} does not divide '
                f'number-of-datasets {data_gen_config.n_datasets}'
            )

        for data_batch_idx in range(n_batches):
            batch_simpson_dirs = rel_simpson_data_dirs[
                data_batch_idx * batch_size: (data_batch_idx + 1) * batch_size
            ]
            batch_non_simpson_dirs = rel_non_simpson_data_dirs[
                data_batch_idx * batch_size: (data_batch_idx + 1) * batch_size
            ]
            logger.info(
            f'Batch {data_batch_idx}: Using {len(batch_simpson_dirs)} simpson and '
            f'{len(batch_non_simpson_dirs)} non-simpson datasets'
        )

            rel_data_dirs = (batch_simpson_dirs + batch_non_simpson_dirs)
            rel_data_dir_strs = ','.join(rel_data_dirs)

            n_runs = len(rel_data_dirs) * len(Cs)
            logger.info(f'Starting sweep over {n_runs} in batch')
            for fit_intercept in ['true','false']:
                subprocess.run([
                    'poetry', 'run', 'python', 'main.py',
                    '--multirun',
                    regularization_overwrite,
                    f'data_version_folder={rel_data_dir_strs}',
                    f'data_batch_idx={data_batch_idx}',
                    f'clf.kwargs.fit_intercept={fit_intercept}',
                    '+hydra.job.chdir=True',
                    '+hydra.sweeper.max_batch_size=10',
                ])


if __name__ == '__main__':
    """
    Usage examples:

    poetry run python multirun-script.py \
        --C_start_power_reg_0 -7 --C_stop_power_reg_0 3 --n_Cs_reg_0 10 \
        --C_start_power_reg_1 -7 --C_stop_power_reg_1 3 --n_Cs_reg_1 10 \
        --C_start_power_reg_2 -7 --C_stop_power_reg_2 3 --n_Cs_reg_2 10 \
        --rel_data_parent outputs/data-2024-04-23/15-53-11

    """
    import argparse
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description='Multirun script for model-fitting')

    # Add arguments
    parser.add_argument('--C_start_power_reg_0', type=int, help='Regularization parameter C power of 10 start')
    parser.add_argument('--C_stop_power_reg_0', type=int, help='Regularization parameter C power of 10 start')
    parser.add_argument('--n_Cs_reg_0', type=int, help='Number of evenly log spaced parameters')
    parser.add_argument('--C_start_power_reg_1', type=int, help='Regularization parameter C power of 10 start')
    parser.add_argument('--C_stop_power_reg_1', type=int, help='Regularization parameter C power of 10 start')
    parser.add_argument('--n_Cs_reg_1', type=int, help='Number of evenly log spaced parameters')
    parser.add_argument('--C_start_power_reg_2', type=int, help='Regularization parameter C power of 10 start')
    parser.add_argument('--C_stop_power_reg_2', type=int, help='Regularization parameter C power of 10 start')
    parser.add_argument('--n_Cs_reg_2', type=int, help='Number of evenly log spaced parameters')
    parser.add_argument('--rel_data_parent', type=str, help='Data parent folder relative to repo root')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        args.C_start_power_reg_0, args.C_stop_power_reg_0, args.n_Cs_reg_0,
        args.C_start_power_reg_1, args.C_stop_power_reg_1, args.n_Cs_reg_1,
        args.C_start_power_reg_2, args.C_stop_power_reg_2, args.n_Cs_reg_2,
        args.rel_data_parent
    )
