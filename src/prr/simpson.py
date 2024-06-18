from typing import Tuple, Dict
from math import prod, copysign

import numpy as np

import xarray as xr


def sample_contingency_table(
    target_sample_size: int, dims: Tuple, coords: Dict, random_generator
):
    """Sample from contingency tables.
    Notes:
    * Due to rounding when scaling up from probability simplex, actual sample size
    may not match target one
    * Not optimized for performance
    """
    n_dims = [len(values) for values in coords.values()]
    alpha = np.ones(prod(n_dims))

    probs = random_generator.dirichlet(alpha, size=1)
    # Scale-up probs to sum to sample size, and then round
    counts = np.round(target_sample_size * probs).astype(int)
    # Reshape
    counts = counts.reshape(n_dims)

    res = xr.DataArray(counts, dims=dims, coords=coords, name='counts')
    return res



def is_simpson(trend_report: Dict, exposure_name: str):
    """TODO refactor, including some optimization"""
    # check if subpopulations exhibit same trend else not (strict) simpson
    exposure_trend_name = exposure_name + '_trend'
    sub_population_trends = []
    for sub_population_report in trend_report['sub_populations']:
        trend = sub_population_report['trend'][exposure_trend_name]
        # trend_sign = math.copysign(1, trend)
        sub_population_trends.append(trend)
    sub_population_trends = np.array(sub_population_trends)
    # Non-simpson if a trend == 0 by (our) definition
    total_trend = trend_report['total_population'][exposure_trend_name]
    zero_subpopulation_trends = sub_population_trends == 0
    if np.any(zero_subpopulation_trends) or total_trend == 0:
        res = False
    else:
        trends_positive = sub_population_trends > 0
        if trends_positive.sum() == 0:
            sub_trends = 'negative'
        elif trends_positive.sum() == len(sub_population_trends):
            sub_trends = 'positive'
        else:
            sub_trends = None
            # At least one trend of different sign than rest, can't be simpson
            res = False

        if sub_trends:
            total_trend_sign = copysign(1, total_trend)
            if total_trend_sign == 1:
                total_trend = 'positive'
            elif total_trend_sign == -1:
                total_trend = 'negative'

            res = sub_trends != total_trend

    return res