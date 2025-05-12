"""Tests for proj_lib.py functions"""
import pytest

from proj_lib import geodetic_to_utm, utm_to_geodetic, UTMZone


@pytest.mark.parametrize(
    "lon, lat, expected_x, expected_y, expected_zone",
    [
        (-61.193534, 41.564028, 650627.09, 4602948.28, '20T'),
        (135.694428, 22.334708, 571511.27, 2470039.89, '53Q'),
    ],
)
def test_geodetic_to_utm(
    lon: float,
    lat: float,
    expected_x: float,
    expected_y: float,
    expected_zone: str,
) -> None:
    """Assert geodetic_to_utm projection is correct."""
    x, y, utm = geodetic_to_utm((lon, lat))

    assert x == pytest.approx(expected_x, abs=0.1)
    assert y == pytest.approx(expected_y, abs=0.1)
    assert utm.zone == expected_zone


@pytest.mark.parametrize(
    "x, y, utm_zone, expected_lon, expected_lat",
    [
        (650627.09, 4602948.28, '20T', -61.193534, 41.564028),
        (571511.27, 2470039.89, '53Q', 135.694428, 22.334708),
    ],
)
def test_utm_to_geodetic(
    x: float,
    y: float,
    utm_zone: str,
    expected_lon: float,
    expected_lat: float,
) -> None:
    """Assert geodetic_to_utm projection is correct."""
    utm_zone = UTMZone.from_str(utm_zone)
    lon, lat = utm_to_geodetic((x, y), utm_zone)

    assert lon == pytest.approx(expected_lon, abs=0.1)
    assert lat == pytest.approx(expected_lat, abs=0.1)
