from __future__ import annotations

from typing import Iterable, TypeAlias

import numpy as np

from mathlib import arcsin, arctan, cos, sin, sec, sech, tan
from util import round_return


Point: TypeAlias = tuple[float, float]


class TMZone:

    a = 6378000
    k_0 = 0.9996

    def __init__(
        self,
        central_longitude: float,
        central_latitude: float,
        convergence: float | None = None,
    ) -> None:
        self.central_longitude = central_longitude
        self.central_latitude = central_latitude

        self.convergence = convergence
        if self.convergence is None:
            self.convergence = self.calc_convergence(
                (central_longitude, central_latitude))

        self.k = self.calc_k()
        self.central_x, self.central_y = self.calc_xy(
            (central_longitude, central_latitude))

    @classmethod
    def from_bbox(cls, points: list[Point]) -> TMZone:
        lons, lats = zip(*points)

        central_lon = np.mean(lons).item()
        central_lat = np.mean(lats).item()

        return TMZone(central_lon, central_lat)

    def calc_convergence(self, point: Point) -> float:
        """Compute the convergence angle given a longitude and latitude
        coordinate pair.

        Parameters
        ----------
        point: Point
            Longitude and latitude coordiante pair, in decimal degrees.

        Returns
        -------
        float
            Convergence angle, in degrees.
        """
        lon, lat = point
        c_rad = np.arctan(tan(lon - self.central_longitude) * sin(lat))

        return np.degrees(c_rad).item()

    @round_return(6)
    def calc_k(self) -> float:
        """Calculate the point scale for the central longitude & latitude."""
        sin2 = sin(self.central_longitude) ** 2
        cos2 = cos(self.central_latitude) ** 2

        return self.k_0 / np.sqrt(1 - sin2 * cos2)

    @round_return(8)
    def calc_lonlat(
        self,
        points: Point | Iterable[Point],
    ) -> Point | Iterable[Point]:
        """Calculate longitude & latitude given Cartesian x- and
        y-coordinates.

        Parameters
        ----------
        x: float
        y: float

        Returns
        -------
        tuple[float, float]
        """
        def _calc_lonlat(x, y) -> Point:
            lon = arctan(
                np.sinh(x / (self.k_0 * self.a)) * sec(y / (self.k_0 * self.a)))

            lat = arcsin(
                sech(x / (self.k_0 * self.a)) * sin(y / (self.k_0 * self.a)))

            return lon.item(), lat.item()

        if isinstance(points, tuple):
            return _calc_lonlat(*points)

        return [_calc_lonlat(*x) for x in points]

    @round_return(8)
    def calc_xy(
        self,
        points: Point | Iterable[Point],
    ) -> Point | Iterable[Point]:
        """Calculate the Cartesian coordinates for the given longitude
        and latitude coordinates.

        Parameters
        ----------
        lon: float
        lat: float

        Returns
        -------
        tuple[float, float]
        """
        def _calc_xy(lon, lat) -> Point:
            numer = 1 + (sin(lon) * cos(lat))
            denom = 1 - sin(lon) * cos(lat)

            x = 0.5 * self.k_0 * self.a * np.log(numer / denom)
            y = self.k_0 * self.a * arctan(sec(lon) * tan(lat))

            return x.item(), y.item()

        if isinstance(points, tuple):
            return _calc_xy(*point)

        return [_calc_xy(*x) for x in points]

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return (
            f'<{self.__class__.__name__}'
            f' ({self.central_longitude}, {self.central_latitude})>')


point = (-80.026280, 26.219869)
zone = TMZone(*point)
