"""Custom transverse mercator projection."""
from __future__ import annotations
from typing import List, Tuple, Union

from pyproj import CRS, Transformer


Point = Tuple[float, float]


class TMTransformer:
    """Class for transforming geodetic longitude & latitude coordinates
    to a custom transverse mercator projection and vice versa."""

    def __init__(
        self,
        lon_0: float,
        lat_0: float = 0.0,
        k_0: float = 0.9996,
        ellps: str = 'WGS84',
        x_0: float = 0.0,
        y_0: float = 0.0,
    ):
        """Instantiate a new TMTransformer instance.

        Parameters
        ----------
        lon_0: float
            Central meridian/longitude of origin.
        lat_0: float, optional
            Latitude of origin. Default is 0.0.
        k_0: float, optional
            Scale factor. Default is 0.9996.
        ellps: str, optional
            Name of pyproj ellipsoid function. Default is "WGS84".
        x_0: float, optional
            False easting, in meters. Default is 0.0.
        y_0: float, optional
            False northing, in meters. Default is 0.0.
        """
        self.lon_0 = lon_0
        self.lat_0 = lat_0
        self.k_0 = k_0
        self.ellps = ellps
        self.x_0 = x_0
        self.y_0 = y_0
        self.crs = {
            'tmerc': CRS.from_proj4(self._get_proj_str()),
            'geodetic': CRS.from_epsg(4326),
        }
        self.transformers = {
            'to_tmerc': Transformer.from_crs(
                self.crs['geodetic'], self.crs['tmerc'], always_xy=True),
            'from_tmerc': Transformer.from_crs(
                self.crs['tmerc'], self.crs['geodetic'], always_xy=True),
        }

    def fwd(
        self,
        coords: Union[Point, List[Point]],
    ) -> Union[Point, List[Point]]:
        """Transform coordinates from geodetic to transverse mercator.

        Parameters
        ----------
        coords: tuple[float, float] or list[tuple[float, float]]
            Longitude and latitude coordinates to transform,
            in decimal degrees.

        Returns
        -------
        tuple[float, float] or list[tuple[float, float]]
            Transformed transverse mercator coordinates.
        """
        return self.transformers['to_tmerc'].transform(*self._get_xy(coords))

    def inv(
        self,
        coords: Union[Point, List[Point]],
    ) -> Union[Point, List[Point]]:
        """Transform coordinates from transverse mercator to geodetic.

        Parameters
        ----------
        coords: tuple[float, float] or list[tuple[float, float]]
            Transverse mercator x- and y-coordinates to transform to
            geodetic longitude & latitude.

        Returns
        -------
        tuple[float, float] or list[tuple[float, float]]
            Transformed geodetic longitude & latitude coordinates.
        """
        return self.transformers['from_tmerc'].transform(*self._get_xy(coords))

    def _get_proj_str(self) -> str:
        """Construct the transverse mercator proj CRS string."""
        return (f'+proj=tmerc +lon_0={self.lon_0} +k_0={self.k_0}'
                f' +x_0={self.x_0} +y_0={self.y_0} +ellps={self.ellps}')

    def _get_xy(
        self,
        coords: Union[Point, List[Point]],
    ) -> Union[Point, Tuple[List[float], List[float]]]:
        """Split coordinate pair(s) into x- and y-coordinates."""
        try:
            x, y = list(zip(*coords))
        except TypeError:
            x, y = coords

        return x, y

    @classmethod
    def from_lonlat(cls, lon: float, lat: float, **kwargs) -> TMTransformer:
        """Return a TMTransformer for the given geodetic longitude
        & latitude coordinates.

        Parameters
        ----------
        lon: float
            Central meridian/longitude of origin.
        lat: float
            Latitude of origin.
        kwargs: str
            Keyword args to pass to constructor.

        Returns
        -------
        TMTransformer
        """
        return TMTransformer(lon, lat_0=lat, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f'<{self.__class__.__name__} ({self.lon_0}, {self.lat_0})>'


if __name__ == '__main__':
    point = (-121.650942,  32.815256)

    xfr = TMTransformer.from_lonlat(*point)

    x, y = xfr.fwd(point)
    lon, lat = xfr.inv((x, y))
