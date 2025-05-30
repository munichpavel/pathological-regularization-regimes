import pytest

import numpy as np
import xarray as xr

from prr.trends import compute_margin, make_a_trend_report, make_trend_reports


@pytest.mark.parametrize(
    'a_data_array,non_margin_sel,expected',
    [
        (
            xr.DataArray(
                [
                    [6, 8],
                    [3, 1]
                ], dims=('recovered', 'treated'),
                coords=dict(recovered=[0, 1], treated=[0, 1])
            ), dict(treated=1),
            np.array(8. + 1.)
        ),
        (
            xr.DataArray(
                [[
                    [6, 8],
                    [3, 1]

                ], [
                    [4, 22],
                    [27, 9]
                ]],
                dims=("recovered", "gender", "treated"),
                coords=dict(recovered=[0, 1], gender=[0, 1], treated=[0, 1])
            ), dict(treated=1),
            np.array(8. + 1. + 22. + 9.)
        ),
        (
            xr.DataArray(
                [[
                    [6, 8],
                    [3, 1]

                ], [
                    [4, 22],
                    [27, 9]
                ]],
                dims=("recovered", "gender", "treated"),
                coords=dict(recovered=[0, 1], gender=[0, 1], treated=[0, 1])
            ), dict(treated=1, gender=1),
            np.array(1. + 9.)
        ),
        (
            xr.DataArray(
                [[
                    [1, 2],
                    [3, 4]

                ], [
                    [5, 6],
                    [7, 8]
                ]],
                dims=("recovered", "gender", "treated"),
                coords=dict(recovered=[0, 1], gender=[0, 1], treated=[0, 1])
            ), dict(),
            np.array(8. * 9. / 2.)
        ),
    ]
)
def test_compute_margin(a_data_array, non_margin_sel, expected):
    res = compute_margin(a_data_array, non_margin_sel)
    assert res == expected


@pytest.mark.parametrize(
    'scores,expected',
    [
        (
            xr.DataArray(
                [42, 42], dims=("gender"), coords={"gender": [0, 1]}
            ),
            dict(gender_trend=0)
        ),
        (
            xr.DataArray(
                [0.4, 0.2], dims=("gender"), coords={"gender": [0, 1]}
            ),
            dict(gender_trend=0.2)
        ),
        (
            xr.DataArray(
                [0.2, 0.4], dims=("gender"), coords={"gender": [0, 1]}
            ),
            dict(gender_trend=-0.2)
        ),
        (
            xr.DataArray(
                [-4, -6], dims=("gender"), coords={"gender": [0, 1]}
            ),
            dict(gender_trend= 2)
        )
    ]
)
def test_make_a_trend_report(scores, expected):
    res = make_a_trend_report(scores)

    assert res == expected


@pytest.mark.parametrize(
    'contingency_probabilities,population_group,expected',
    [
        (
            xr.DataArray(
                np.array([
                    [0.4, 0.1],
                    [0.2, 0.8]]
                ),
                dims=('gender', 'occupation'),
                coords=dict(gender=[0, 1], occupation=[0, 1])
            ),
            'occupation',
            [
                {
                    'population_group': 'occupation',
                    'population_value': 0,
                    'trend': {'gender_trend': pytest.approx(0.2)}
                },
                {
                    'population_group': 'occupation',
                    'population_value': 1,
                    'trend': {'gender_trend': pytest.approx(-0.7)}
                }
            ]
        ),
        (
            xr.DataArray(
                np.array([
                    [4, 1],
                    [2, 8]]
                ),
                dims=('gender', 'occupation'),
                coords=dict(gender=[0, 1], occupation=[0, 1])
            ),
            'occupation',
            [
                {
                    'population_group': 'occupation',
                    'population_value': 0,
                    'trend': {'gender_trend': pytest.approx(2.)}
                },
                {
                    'population_group': 'occupation',
                    'population_value': 1,
                    'trend': {'gender_trend': pytest.approx(-7.)}
                }
            ]
        ),
        (
            xr.DataArray(
                np.array([0.2, 0.1]),
                dims=('gender',),
                coords=dict(gender=[0, 1])
            ),
            '',
            [{'gender_trend': pytest.approx(0.1)}]
        )

    ]
)
def test_make_trend_reports(contingency_probabilities,population_group,expected):
    res = make_trend_reports(
        contingency_probabilities, population_group
    )
    assert res == expected
