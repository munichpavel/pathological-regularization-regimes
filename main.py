import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf


from prr.utils import (
    convert_to_categorical,
    get_object_from_module,
    generate_log_spaced_grid
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('simpsons-paradox-main')


# TODO (maybe) put these in config
TARGET_FIELD_NAME = 'default'
EXPOSURE_FIELD_NAME = 'gender'
NON_EXPOSURE_FIELD_NAME = 'occupation'


@hydra.main(version_base=None, config_path="conf", config_name='base-model-fitting')
def main(cfg: DictConfig) -> None:

    datadir = Path(os.environ['REPO_ROOT']) / cfg.data_version_folder
    logger.info("Loading data")
    df = pd.read_csv(datadir / 'default.csv')
    n_records = df.shape[0]
    logger.info(f"Dataset has {n_records} records")

    data_categories = dict(cfg.encoding.label_mapping_values)
    data_categories[TARGET_FIELD_NAME] = [0, 1]
    df = convert_to_categorical(df, data_categories)

    feature_fields = list(cfg.encoding.label_mapping_values.keys())
    logger.debug(f'Using feature fields {feature_fields}')

    X = df.loc[:, feature_fields].values
    y = df[TARGET_FIELD_NAME]

    logger.info('Fitting model')
    Clf = get_object_from_module(parent_module_name=cfg.clf.parent_module, object_name=cfg.clf.class_name)
    C_grid = generate_log_spaced_grid(cfg.clf.C_grid).tolist()
    model_params = []
    for C_value in C_grid:
        model_kwargs = OmegaConf.to_container(cfg.clf.kwargs)
        model_kwargs['C'] = C_value  # type: ignore
        clf = Clf(**model_kwargs)  # type: ignore
        clf.fit(X, y)

        # check for convergence
        if clf.n_iter_[0] == clf.max_iter:
            logger.info('Solver did not converge, skipping')
            model_params.append(
                dict(C=C_value, b=np.nan, w_0=np.nan, w_1=np.nan)
            )
        else:

            b = clf.intercept_
            w = clf.coef_
            model_params.append(
                dict(C=C_value, b=b[0], w_0=w[0,0], w_1=w[0,1])
            )

    model_params_df = pd.DataFrame(model_params)
    model_params_df.to_csv('model-paramses.csv', index=False)

if __name__ == '__main__':
    main()
