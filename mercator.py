"""This module contains classes for transforming between geodetic, transverse,
and universal transverse mercator projections."""
from __future__ import annotations

from typing import Iterable, TypeAlias

import numpy as np
import utm

from mathlib import arcsin, arctan, cos, sin, sec, sech, tan
from util import round_return


Point: TypeAlias = tuple[float, float]


class TMZone:
    """A Transverse Mercator zone."""

    a = 6378000
    k_0 = 0.9996

    def __init__(
        self,
        central_longitude: float,
        central_latitude: float,
        convergence: float | None = None,
    ) -> None:
        """Instantiate a new TMZone.

        Parameters
        ----------
        central_longitude: float
            Central longitude of the projection.
        central_latitude: float
            Central latitude of the projection.
        convergence: float, optional
            Convergence angle.
        """
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
        """Instantiate a new TMZone.

        Parameters
        ----------
        TODOs
        """
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
        """Calculate longitude & latitude given Cartesian x- and y-coordinates.

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
            return _calc_xy(*points)

        return [_calc_xy(*x) for x in points]

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return (
            f'<{self.__class__.__name__}'
            f' ({self.central_longitude}, {self.central_latitude})>')


class UTMZone:
    """A Universal Transverse Mercator Zone."""

    def __init__(
        self,
        number: int,
        letter: str,
        convergence: float | None = None,
    ):
        """Instantiate a new UTMZone object.

        Paramters
        ---------
        number: int
            UTM zone number.
        letter: str
            UTM zone letter.
        convergence: float, optional
            Convergence angle for an arbitrary longitude.
        """
        self.number = number
        self.letter = letter
        self.convergence = convergence
        self.central_longitude = utm.zone_number_to_central_longitude(number)
        self.central_latitude = utm.zone_letter_to_central_latitude(letter)

    @property
    def zone(self) -> str:
        """Return the zone number of letter as a string."""
        return f'{self.number}{self.letter}'

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

        return np.degrees(c_rad)

    def set_convergence(self, point: Point):
        """Set thje instance convergence angle.

        Parameters
        ----------
        point: Point
            Longitude and latitude coordiante pair, in decimal degrees.

        Returns
        -------
        None
        """
        self.convergence = self.calc_convergence(point)

    def geodetic_to_utm(
        self,
        points: Point | Iterable[Point],
    ) -> Point | list[Point]:
        """Project geodetic longitude & latitude coordinate pairs to UTM.

        Parameters
        ----------
        point: Point or iterable of Points
            Longitude & latitude coordinate pairs to project.

        Returns
        -------
        list of Point
            x- and y-coordinate pairs
        """
        lons, lats = self._format_coords(points)
        x, y, _, _ = utm.from_latlon(lats, lons)

        if isinstance(x, float):
            return (x, y)

        return list(zip(x.tolist(), y.tolist()))

    def utm_to_geodetic(
        self,
        points: Point | Iterable[Point],
    ) -> Point | list[Point]:
        """Project UTM x- and y-coordinates to geodetic longitude & latitude.

        Parameters
        ----------
        point: Point or iterable of Points
            Easting and northing coordinate pairs to project.

        Returns
        -------
        list of Point
            Longitude and latitude coordinate pairs.
        """
        x, y = self._format_coords(points)

        lats, lons = utm.to_latlon(x, y, self.number, self.letter)

        if isinstance(lats, float):
            return (lons, lats)

        return list(zip(lons.tolist(), lats.tolist()))

    @classmethod
    def from_lonlat(cls, points: Point | Iterable[Point]) -> UTMZone:
        """Project geodetic longitude & latitude coordinate pairs to UTM.

        Parameters
        ----------
        point: Point or iterable of Points
            Longitude & latitude coordinate pairs to project.

        Returns
        -------
        list of Point
            x- and y-coordinate pairs
        UTMZone
            UTM zone metadata.
        """
        lons, lats = cls._format_coords(points)

        zone = UTMZone(
            utm.latlon_to_zone_number(lats, lons),
            utm.latitude_to_zone_letter(lats),
        )
        zone.set_convergence(points)

        return zone

    @classmethod
    def _format_coords(cls, coords: Point | Iterable[Point]) -> tuple:
        """Format coordinate pairs into two numpy arrays."""
        if isinstance(coords, list) or isinstance(coords, tuple):
            points = np.array(coords)

        return points.T

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f'<{self.__class__.__name__} {self.zone}>'
