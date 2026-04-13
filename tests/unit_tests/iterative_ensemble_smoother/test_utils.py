import numpy as np
import pytest

from iterative_ensemble_smoother.utils import calc_rho_for_2d_grid_layer


def test_that_calc_rho_for_2d_grid_layer_ignores_obs_outside_the_grid():
    """Test that observations positioned outside the grid with small ranges
    result in zero RHO values for all grid cells
    """
    nx, ny = 3, 3
    xinc, yinc = 1.0, 1.0

    obs_xpos = np.array([1.5, 100.0])  # Second obs is far outside.
    obs_ypos = np.array([1.5, 100.0])

    obs_main_range = np.array([1.0, 1.0])
    obs_perp_range = np.array([1.0, 1.0])
    obs_anisotropy_angle = np.array([0.0, 0.0])

    rho = calc_rho_for_2d_grid_layer(
        nx=nx,
        ny=ny,
        xinc=xinc,
        yinc=yinc,
        obs_xpos=obs_xpos,
        obs_ypos=obs_ypos,
        obs_main_range=obs_main_range,
        obs_perp_range=obs_perp_range,
        obs_anisotropy_angle=obs_anisotropy_angle,
    )

    assert rho.shape == (3, 3, 2)

    rho_inside = rho[:, :, 0]
    assert np.any(rho_inside > 0), "Observation inside the grid should have nonzero rho"

    rho_outside = rho[:, :, 1]
    assert np.all(rho_outside == 0.0), (
        "Observation outside the grid should have all zero rho values"
    )


def test_that_calc_rho_for_2d_grid_layer_validates_observation_lengths():
    with pytest.raises(
        ValueError,
        match="Number of coordinates must match number of observations",
    ):
        calc_rho_for_2d_grid_layer(
            nx=3,
            ny=3,
            xinc=1.0,
            yinc=1.0,
            obs_xpos=np.array([1.5]),
            obs_ypos=np.array([1.5, 2.5]),
            obs_main_range=np.array([1.0]),
            obs_perp_range=np.array([1.0]),
            obs_anisotropy_angle=np.array([0.0]),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nx": 0}, "`nx` must be positive"),
        ({"ny": -1}, "`ny` must be positive"),
        ({"xinc": 0.0}, "`xinc` must be positive"),
        ({"yinc": -0.5}, "`yinc` must be positive"),
        (
            {"obs_main_range": np.array([0.0])},
            "All main-range values for all observations must be positive",
        ),
        (
            {"obs_perp_range": np.array([-1.0])},
            "All perpendicular-range values for all observations must be positive",
        ),
    ],
)
def test_that_calc_rho_for_2d_grid_layer_validates_inputs(kwargs, message):
    args = {
        "nx": 3,
        "ny": 3,
        "xinc": 1.0,
        "yinc": 1.0,
        "obs_xpos": np.array([1.5]),
        "obs_ypos": np.array([1.5]),
        "obs_main_range": np.array([1.0]),
        "obs_perp_range": np.array([1.0]),
        "obs_anisotropy_angle": np.array([0.0]),
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        calc_rho_for_2d_grid_layer(**args)
