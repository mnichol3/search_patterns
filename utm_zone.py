from __future__ import annotations

from typing import Iterable, TypeAlias

import numpy as np
import utm

from mathlib import sin, tan

Point: TypeAlias = tuple[float, float]


class UTMZone:

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
        return f'<UTMZone {self.zone}>'
