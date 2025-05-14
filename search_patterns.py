from collections.abc import Callable
from copy import deepcopy

import pandas as pd
from pyproj import Geod
from shapely import LineString, Polygon

from cartesian import azimuth, calc_distance, calc_fwd
from mathlib import M_2_NMI
from util import round_return
from utm_zone import UTMZone


class BaseSearchPattern:
    """Search pattern base class."""

    def __init__(self, csp: tuple[float, float]):
        """BaseSearchPattern constructor.

        Parameters
        ----------
        csp: tuple of float, float
            X- and y-coordinates of the commence search point.
        """
        self.csp = csp
        self.waypoints = None

    def to_dataframe(
        self,
        transformer: Callable | None = None,
    ) -> pd.DataFrame:
        """Return the search pattern as a dataframe.

        Parameters
        ----------
        utm_zone: UTMZone, optional
            If passed, the coordinates will be projected from transverse
            mercator to geodetic longitude & latitude and distances will be
            given in nautical miles.
            Default is None, meaning the coordinates will stay in TM and
            distances will be given in meters.

        Returns
        -------
        pandas DataFrame
        """
        def _eucl_dist(p1: tuple[float, float], p2: tuple[float, float]):
            return calc_distance(p1, p2)

        @round_return(2)
        def _geod_dist(p1: tuple[float, float], p2: tuple[float, float]):
            return geod.inv(*p1, *p2)[-1] * M_2_NMI

        if len(self.waypoints) == 0:
            raise ValueError(f'Waypoints list is empty.')

        columns = [
            'lon',
            'lat',
            'distance',
            'cum_distance',
        ]

        geod = None
        dist_func = _eucl_dist
        waypoints = deepcopy(self.waypoints)

        if transformer:
            waypoints = transformer(waypoints)
            geod = Geod(ellps='WGS84')
            dist_func = _geod_dist

        leg_len = 0
        cum_dist = 0
        prev_waypoint = waypoints[0]

        waypoints[0] = list(prev_waypoint) + [leg_len, cum_dist]

        for i, waypoint in enumerate(waypoints[1:]):
            leg_len = dist_func(prev_waypoint, waypoint)
            cum_dist += leg_len
            waypoints[i + 1] = list(waypoint) + [leg_len, cum_dist]
            prev_waypoint = waypoint

        return pd.DataFrame(waypoints, columns=columns)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        lon = round(self.csp[0], 6)
        lat = round(self.csp[1], 6)
        return f'<{self.__class__.__name__} ({lon}, {lat})>'


class SectorSearch(BaseSearchPattern):
    """Class for generating a sector search pattern."""

    def __init__(
        self,
        csp: tuple[float, float],
        convergence: float = 0.0,
    ):
        """Instantiate a SectorSearch object.

        Parameters
        ----------
        csp: tuple of float, float
            X- and y-coordinates of the commence search point.
        radius: int
            Pattern radius, unitless.
        orientation: int
            Pattern orientatin, in degrees.
        """
        super().__init__(csp)
        self.convergence = convergence

    def run(
        self,
        radius: int,
        orientation: int,
        n_patterns: int = 1,
    ) -> list[tuple[float, float]]:
        """Generate the sector search pattern.

        Parameters
        ----------
        n_patterns : int, optional
            Number of patterns to generate, by default 1.

        Returns
        -------
        list[tuple[float, float]]
            Pattern waypoints.
        """
        n_patt = 0
        waypoints = []
        track = azimuth(orientation - self.convergence)

        while n_patt < n_patterns:
            waypoints.extend(self._gen_pattern(self.csp, track, radius))
            track = azimuth(track + 30)
            n_patt += 1

        self.waypoints = waypoints

        return waypoints

    def _gen_pattern(
        self,
        csp: tuple[float, float],
        track: int,
        radius: int,
    ) -> list[tuple[float, float]]:
        """_summary_

        Parameters
        ----------
        csp : tuple[float, float]
            X- and y-coordinates of the commence search point.
        track : int
            Pattern leg track.

        Returns
        -------
        list[tuple[float, float]]
            Pattern waypoints.
        """
        def _turn(trk: int) -> int:
            return azimuth(trk + 120)

        prev_point = calc_fwd(csp, track, radius)
        waypoints = [self.csp, prev_point]
        track = _turn(track)
        n_leg = 1

        while n_leg < 6:
            curr_len = radius * 2 if n_leg % 2 == 0 else radius
            curr_point = calc_fwd(prev_point, track, curr_len)
            waypoints.append(curr_point)
            prev_point = curr_point
            track = _turn(track)
            n_leg += 1

        waypoints.append(calc_fwd(prev_point, track, radius))

        return waypoints

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        lon = round(self.csp[0], 6)
        lat = round(self.csp[1], 6)
        return f'<SectorSearch ({lon}, {lat})>'


class ParallelTrackSearch(BaseSearchPattern):
    """Class for generating a parallel track search pattern."""

    def __init__(
        self,
        poly: Polygon,
        csp: tuple[float, float],
        convergence: float = 0.0,
    ):
        """Instantiate a ParallelTrackSearch object.

        Parameters
        ----------
        csp: tuple of float, float
            X- and y-coordinates of the commence search point.
        radius: int
            Pattern radius, unitless.
        orientation: int
            Pattern orientatin, in degrees.
        """
        super().__init__(csp)
        self.poly = poly
        self.convergence = convergence

    def run(
        self,
        first_course: float,
        creep: float,
        track_spacing: float,
    ) -> list[tuple[float, float]]:
        """Generate the parallel track search pattern.

        Parameters
        ----------
        None

        Returns
        -------
        list[tuple[float, float]]
            Pattern waypoints.
        """
        def _turn(crs: float, by: float) -> float:
            return (crs + by) % 360.

        waypoints = []
        vector_len = int(self.poly.length / 2)
        leg_num = 0

        # Step inside and backwards to get 2 intersection points with the
        # polygon
        curr_pt = calc_fwd(self.csp, creep, track_spacing / 2)
        curr_pt = calc_fwd(
            curr_pt, _turn(first_course, -180), track_spacing / 2)

        fwd_pt = calc_fwd(curr_pt, first_course, vector_len)
        intersection = self.poly.intersection(LineString([curr_pt, fwd_pt]))

        while not intersection.is_empty:
            leg_points = list(
                zip(*[x.tolist() for x in intersection.coords.xy]))

            if leg_num % 2 == 0:
                waypoints.extend(leg_points)
            else:
                waypoints.extend(leg_points[::-1])

            curr_pt = calc_fwd(curr_pt, creep, track_spacing)
            fwd_pt = calc_fwd(curr_pt, first_course, vector_len)
            intersection = self.poly.intersection(LineString([curr_pt, fwd_pt]))

            leg_num += 1

        self.waypoints = waypoints

        return waypoints
