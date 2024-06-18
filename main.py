import os
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig


from prr.utils import (
    convert_to_categorical,
    get_object_from_module,
    make_feature_combination_array,
    make_feature_combination_score_array
)

from prr.trends import make_trend_reports


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
    clf = Clf(**cfg.clf.kwargs)
    clf.fit(X, y)

    # check for convergence
    if clf.n_iter_[0] == clf.max_iter:
        logger.info('Solver did not converge, skipping')
    else:

        b = clf.intercept_
        w = clf.coef_.tolist()
        coefs = dict(b=b.tolist(), w=w)

        logger.debug(f"Writing b, w to disk: {coefs}")
        with open(f'coefs.json', 'w') as fp:
            json.dump(coefs, fp=fp, indent=2)


        logger.debug("Evaluating on feature combinations")
        feature_combination_array = make_feature_combination_array(
            label_mapping_values=dict(cfg.encoding.label_mapping_values)
        )

        scores = clf.predict_proba(feature_combination_array)
        prob_default = scores[:, [1]]  # Class 1 is default, class 0 no-default


        feature_combination_scores = pd.DataFrame(
            np.concatenate([feature_combination_array, prob_default], axis=1),
            columns=feature_fields + ['default_score']
        )
        for feature in feature_fields:
            feature_combination_scores[feature] = feature_combination_scores[feature].astype(int)
        feature_combination_scores.to_csv(f'feature_probabilities.csv', index=False)

        # Convert flat probabilities to xarray for more robust accessing
        feature_combination_contingency = make_feature_combination_score_array(
            feature_combinations=feature_combination_scores[feature_fields],
            scores=feature_combination_scores['default_score']
        )

        sub_population_trend_reports = make_trend_reports(
            feature_combination_contingency, population_subgroup=NON_EXPOSURE_FIELD_NAME
        )

        logger.debug(f"Writing trend reports {sub_population_trend_reports}")
        with open('model_trend_reports.json', 'w') as fp:
            json.dump(sub_population_trend_reports, fp, indent=2)


if __name__ == '__main__':
    main()
