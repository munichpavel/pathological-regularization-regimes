import pytest

import numpy as np


from prr.simpson import sample_contingency_table, is_simpson


@pytest.mark.parametrize(
    '''
    target_sample_size,
    dims,
    coords,
    random_seed,
    expected_values
    ''',
    [
        (
            1,
            ('default', 'gender', 'occupation'),
            dict(default=[0, 1], gender=[0, 1], occupation=[0, 1]),
            42,
            # Example of target sample size > returned sample size
            # due to numpy rounding behavior
            np.array(
                [[
                    [0, 0],
                    [0, 0]
                ], [
                    [0, 0],
                    [0, 0]

                ]]
            )
        ),
        (
            10,
            ('default', 'gender', 'occupation'),
            dict(default=[0, 1], gender=[0, 1], occupation=[0, 1]),
            42,
            # Example of target sample size == returned sample size
            np.array(
                [[
                    [2, 2],
                    [2, 0]
                ], [
                    [0, 1],
                    [1, 2]

                ]]
            )
        ),
    ]
)
def test_sample_contingency_table(
    target_sample_size,
    dims,
    coords,
    random_seed,
    expected_values
):
    """Using a given random seed for reproducibility"""
    rng = np.random.default_rng(seed=random_seed)

    contingency_table = sample_contingency_table(
        target_sample_size=target_sample_size,
        dims=dims, coords=coords,
        random_generator=rng
    )
    assert (contingency_table.values == expected_values).all()


@pytest.mark.parametrize(
    'trend_report,exposure_name,expected',
    [
          (
            {
                "total_population": {
                    "gender_trend": 1
                },
                "sub_populations": [
                    {
                    "population_group": "occupation",
                    "population_value": 0,
                    "trend": {
                        "gender_trend": 0  # 0 trend -> non-simpson
                    }
                    },
                    {
                    "population_group": "occupation",
                    "population_value": 1,
                    "trend": {
                        "gender_trend": -1
                    }
                    }
                ]
            },
            'gender', False
        ),
        (
            {
                "total_population": {
                    "gender_trend": 0  # 0 trend -> non-simpson
                },
                "sub_populations": [
                    {
                    "population_group": "occupation",
                    "population_value": 0,
                    "trend": {
                        "gender_trend": -1
                    }
                    },
                    {
                    "population_group": "occupation",
                    "population_value": 1,
                    "trend": {
                        "gender_trend": -1
                    }
                    }
                ]
            },
            'gender', False
        ),
        (
            {
                "total_population": {
                    "gender_trend": 1
                },
                "sub_populations": [
                    {
                    "population_group": "occupation",
                    "population_value": 0,
                    "trend": {
                        "gender_trend": 1  # 0 trend -> non-simpson
                    }
                    },
                    {
                    "population_group": "occupation",
                    "population_value": 1,
                    "trend": {
                        "gender_trend": -1
                    }
                    }
                ]
            },
            'gender', False
        ),
        (
            {
                "total_population": {
                    "gender_trend": 1
                },
                "sub_populations": [
                    {
                    "population_group": "occupation",
                    "population_value": 0,
                    "trend": {
                        "gender_trend": -1
                    }
                    },
                    {
                    "population_group": "occupation",
                    "population_value": 1,
                    "trend": {
                        "gender_trend": -1
                    }
                    }
                ]
            },
            'gender', True
        ),
        (
            {
                "total_population": {
                    "gender_trend": 0.13063063063063066
                },
                "sub_populations": [
                    {
                    "population_group": "occupation",
                    "population_value": 0,
                    "trend": {
                        "gender_trend": 0.05238095238095236
                    }
                    },
                    {
                    "population_group": "occupation",
                    "population_value": 1,
                    "trend": {
                        "gender_trend": -0.05280172413793102
                    }
                    }
                ]
            },
            'gender', False
        )

    ]
)
def test_is_simpson(trend_report, exposure_name, expected):
    res = is_simpson(trend_report, exposure_name=exposure_name)

    assert res == expected
