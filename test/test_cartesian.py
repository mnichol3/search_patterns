"""Tests for functions in cartesian.py"""
import pytest

import cartesian


@pytest.mark.parametrize(
    "azi, expected",
    [
        (0, 90),
        (30, 60),
        (60, 30),
        (90, 0),
        (135, 315),
        (150, 300),
        (215, 235),
        (270, 180),
        (315, 135),
        (360, 90),
    ],
)
def test_azimuth_to_uangle(azi: float, expected: float) -> None:
    """Assert azimuth_to_uangle() computes the correct unit angle.

    Parameters
    ----------
    azi: float
        Azimuth
    expected: float
        Expected value.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
    """
    assert cartesian.azimuth_to_uangle(azi) == expected


@pytest.mark.parametrize(
    "point1, point2, expected",
    [
        ((0, 0), (0, 0), 0),
        ((0, 0), (2, 2), 45),
        ((0, 0), (2, 0), 90),
        ((0, 0), (0, -2), 180),
        ((0, 0), (-2, -2), 225),
        ((0, 0), (-2, 0), 270),
        ((0, 0), (-2, 2), 315),
    ],
)
def test_calc_azimuth(
    point1: tuple[float, float],
    point2: tuple[float, float],
    expected: float,
) -> None:
    """Assert calc_azimuth() computes the correct azimuth.

    Parameters
    ----------
    point1: tuple of float, float
        Point 1 x- and y-coordinates.
    point2: tuple of float, float
        Point 2 x- and y-coordinates.
    expected: float
        Expected value.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
    """
    assert cartesian.calc_azimuth(point1, point2) == expected


@pytest.mark.parametrize(
    "origin, azimuth, dist, expected",
    [
        ((0, 0), 0, 5, (0.0, 5.0)),
        ((0, 0), 180, 10, (0.0, -10.0)),
        ((0, 0), 270, 10, (-10.0, 0.0)),
    ],
)
def test_calc_fwd(
    origin: tuple[float, float],
    azimuth: float,
    dist: float,
    expected: tuple[float, float],
) -> None:
    """Assert calc_azimuth() computes the correct azimuth.

    Parameters
    ----------
    origin: tuple of float, float
        Origin x- and y-coordinates.
    azimuth: float
        Azimuth to move along.
    dist: float
        Unitless distance.
    expected: tuple of float, float
        Expected value.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
    """
    assert cartesian.calc_fwd(origin, azimuth, dist) == expected
