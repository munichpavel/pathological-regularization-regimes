"""Example usage
poetry run python generate-data.py +hydra.job.chdir=True
"""

import json
from pathlib import Path
import logging
import warnings
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
from fake_data_for_learning.contingency_tables import (
    generate_fake_data_for_contingency_table
)

from prr.simpson import sample_contingency_table, is_simpson
from prr.trends import make_data_trend_reports
from prr.utils import convert_to_categorical


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='data-generation.yaml')
def main(cfg: DictConfig) -> None:
    config_str = json.dumps(OmegaConf.to_container(cfg), indent=2)
    logger.info(f'Generating data with config\n{config_str}')
    n_datasets = cfg.n_datasets
    random_seed = cfg.random_seed


    for sample_size in cfg.sample_sizes:
        out_dir = Path(f'samples-{sample_size}')
        if out_dir.exists():
            shutil.rmtree(out_dir)

        generate_data(
            n_datasets=n_datasets, sample_size=sample_size,
            random_seed=random_seed,
            out_dir=out_dir
        )


def generate_data(n_datasets: int, sample_size: int, random_seed: int, out_dir: Path) -> None:
    rng = np.random.default_rng(seed=random_seed)

    gender_values = [0, 1]
    occupation_values = [0, 1]
    default_values = [0, 1]

    contingency_dims = ('default', 'gender', 'occupation')
    contingency_coords = dict(
        default=default_values, gender=gender_values, occupation=occupation_values
    )

    simpson_outdir = out_dir / 'simpson'
    non_simpson_outdir = out_dir / 'non-simpson'

    n_simpsons = 0
    n_non_simpsons = 0
    simpson_finished = False
    non_simpson_finished = False

    logger.info(
        f'Start sampling {n_datasets} datasets of size {sample_size}'
        ' for each of simpson and non-simpson'
    )
    log_n_finisheds = 50
    while not simpson_finished or not non_simpson_finished:

        contingency_table = sample_contingency_table(
            target_sample_size=sample_size,
            dims=contingency_dims,
            coords=contingency_coords,
            random_generator=rng

        )
        if contingency_table.sum() == sample_size:
            df = generate_fake_data_for_contingency_table(contingency_table)
            # Convert dtype to categorical for counting
            df = convert_to_categorical(df, categorical_values_dict=contingency_coords)

            data_trend_report = make_data_trend_reports(
                df=df, target_field_name='default', target_field_value=1,
                non_exposure_field_name='occupation'
            )

            # Determine simpson-ness or not
            simpson_flag = is_simpson(data_trend_report, exposure_name='gender')

            if simpson_flag:
                if n_simpsons < n_datasets:

                    simpson_sample_outdir = simpson_outdir / str(n_simpsons)
                    logger.debug(f'Writing simpson sample {n_simpsons} to directory {simpson_sample_outdir}')
                    simpson_sample_outdir.mkdir(parents=True)
                    with open(simpson_sample_outdir  /  'trend-report.json', 'w') as fp:
                        json.dump(data_trend_report, fp, indent=2)

                    with open(simpson_sample_outdir / 'default.csv', 'w') as fp:
                        df.to_csv(fp.name, index=False)
                    n_simpsons += 1
                    if n_simpsons % log_n_finisheds == 0:
                        logger.info(f'{n_simpsons} simpson datasets created of sample-size {sample_size}')
                else:
                    simpson_finished = True
            else:
                if n_non_simpsons < n_datasets:

                    non_simpson_sample_outdir = non_simpson_outdir / str(n_non_simpsons)
                    logger.debug(f'Writing non-simpson sample {n_non_simpsons} to directory {non_simpson_sample_outdir}')
                    non_simpson_sample_outdir.mkdir(parents=True)
                    with open (non_simpson_sample_outdir / 'trend-report.json', 'w') as fp:
                        json.dump(data_trend_report, fp, indent=2)
                    with open(non_simpson_sample_outdir / 'default.csv', 'w') as fp:
                        df.to_csv(fp.name, index=False)
                    n_non_simpsons += 1
                    if n_non_simpsons % log_n_finisheds == 0:
                        logger.info(f'{n_non_simpsons} non-simpson datasets created of sample-size {sample_size}')

                else:
                    non_simpson_finished = True
        else:
            logger.debug(
                f'Discard sample: Size of sampled data {float(contingency_table.sum())} '
                f'does not match sample-size {sample_size}.'
            )


if __name__ == '__main__':
    main()
