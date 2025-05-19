from collections.abc import Callable
from copy import deepcopy
from typing import TypeAlias

import pandas as pd
from pyproj import Geod
from shapely import LineString, Polygon

from cartesian import azimuth, calc_distance, calc_fwd
from dubins import DubinsPath, Turn
from mathlib import M_2_NMI
from util import round_return
from waypoint import Waypoint


Point: TypeAlias = tuple[float, float]


class BaseSearchPattern:
    """Search pattern base class."""

    def __init__(self, csp: Point):
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
        transformer: Callable, optional
            If passed, the coordinates will be projected from transverse
            mercator to geodetic longitude & latitude and distances will be
            given in nautical miles.
            Default is None, meaning the coordinates will stay in TM and
            distances will be given in meters.

        Returns
        -------
        pandas DataFrame
        """
        def _eucl_dist(p1: Point, p2: Point):
            return calc_distance(p1, p2)

        @round_return(2)
        def _geod_dist(p1: Point, p2: Point):
            azi, _ , dist = geod.inv(*p1, *p2)
            return azi, dist * M_2_NMI

        if len(self.waypoints) == 0:
            raise ValueError(f'Waypoints list is empty.')

        columns = [
            'lon',
            'lat',
            'course',
            'distance',
            'cum_distance',
        ]

        rows = []

        dist_func = _eucl_dist
        waypoints = deepcopy(self.waypoints)

        if transformer:
            waypoints = transformer(waypoints)
            geod = Geod(ellps='WGS84')
            dist_func = _geod_dist

        leg_len = 0
        cum_dist = 0
        prev_waypoint = waypoints[0]

        rows.append(prev_waypoint + (-999, leg_len, cum_dist))

        for waypoint in waypoints[1:]:
            crs, leg_len = dist_func(prev_waypoint, waypoint)
            cum_dist += leg_len
            rows.append(waypoint + (crs, leg_len, cum_dist))
            prev_waypoint = waypoint

        return pd.DataFrame(rows, columns=columns)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        lon = round(self.csp[0], 6)
        lat = round(self.csp[1], 6)
        return f'<{self.__class__.__name__} ({lon}, {lat})>'


class ExpandingSquare(BaseSearchPattern):
    """Class for generating an expanding square search pattern."""

    def __init__(self, csp: Point):
        """Instantiate a SectorSearch object.

        Parameters
        ----------
        csp: tuple of float, float
            X- and y-coordinates of the commence search point.
        """
        super().__init__(csp)

    def run(
        self,
        first_course: int,
        d: float,
        num_legs: int = 12,
        turn_dir: int = -1,
    ) -> list[Point]:
        """Generate the sector search pattern.

        Parameters
        ----------
        first_course: int
            Course of the first leg of the pattern.
        d: float
            Initial leg length, in meters.
        num_legs: int, optional
            Number of legs. Default is 12.
        turn_dir: int, optional
            Turn direction. 1 for right, -1 for left. Default is -1.

        Returns
        -------
        list[tuple[float, float]]
            Pattern waypoints.
        """
        n = 0
        curr_d = d
        prev_point = self.csp
        waypoints = [prev_point]
        curr_crs = first_course

        if turn_dir not in [-1, 1]:
            raise ValueError(
                f'turn_dir parameters must be -1 or 1, got "{turn_dir}"')

        while n <= num_legs:
            new_point = calc_fwd(prev_point, curr_crs, curr_d)
            waypoints.append(new_point)
            prev_point = new_point

            curr_crs = azimuth(curr_crs + (turn_dir * 90))

            if n % 2 != 0:
                curr_d += d

            n += 1

        self.waypoints = waypoints

        return waypoints


class SectorSearch(BaseSearchPattern):
    """Class for generating a sector search pattern."""

    def __init__(self, csp: Point):
        """Instantiate a SectorSearch object.

        Parameters
        ----------
        csp: tuple of float, float
            X- and y-coordinates of the commence search point.
        """
        super().__init__(csp)

    def run(
        self,
        first_course: int,
        radius: int,
        n_patterns: int = 1,
    ) -> list[Point]:
        """Generate the sector search pattern.

        Parameters
        ----------
        first_course: int
            Course of the first leg, in degrees.
        radius: float
            Radius, in meters.
        n_patterns: int, optional
            Number of patterns to generate, by default 1.

        Returns
        -------
        list[tuple[float, float]]
            Pattern waypoints.
        """
        n_patt = 0
        waypoints = []
        curr_course = first_course

        while n_patt < n_patterns:
            waypoints.extend(self._gen_pattern(self.csp, curr_course, radius))
            curr_course = azimuth(curr_course + 30)
            n_patt += 1

        self.waypoints = waypoints

        return waypoints

    def _gen_pattern(
        self,
        csp: Point,
        track: int,
        radius: int,
    ) -> list[Point]:
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

    def __init__(self, poly: Polygon, csp: Point):
        """Instantiate a ParallelTrackSearch object.

        Parameters
        ----------
        poly: shapely.Polygon
            OpArea polygon.
        csp: tuple of float, float
            X- and y-coordinates of the commence search point.
        """
        super().__init__(csp)
        self.poly = poly

    def run(
        self,
        first_course: int,
        creep: float,
        track_spacing: float,
        turn_radius: float | None = None,
    ) -> list[Point]:
        """Generate the parallel track search pattern.

        Parameters
        ----------
        first_course: int
            Course of the first leg of the pattern, in degrees.
        creep: int
            Pattern creep direction, in degrees.
        track_spacing: float
            Track spacing, in meters.

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
        true_creep = (first_course + creep) % 360.

        # Step inside and backwards to get 2 intersection points with the
        # polygon
        curr_pt = calc_fwd(self.csp, true_creep, track_spacing / 2)
        curr_pt = calc_fwd(
            curr_pt, _turn(first_course, -180), track_spacing / 2)

        fwd_pt = calc_fwd(curr_pt, first_course, vector_len)
        intersection = self.poly.intersection(LineString([curr_pt, fwd_pt]))

        while not intersection.is_empty:
            leg_points = list(
                zip(*[x.tolist() for x in intersection.coords.xy]))

            if leg_num % 2 == 0:
                waypoints.extend(
                    [Waypoint(*x, first_course) for x in leg_points])
            else:
                waypoints.extend(
                    [Waypoint(*x, first_course + 180.)
                     for x in leg_points][::-1])

            curr_pt = calc_fwd(curr_pt, true_creep, track_spacing)
            fwd_pt = calc_fwd(curr_pt, first_course, vector_len)

            intersection = self.poly.intersection(
                LineString([curr_pt, fwd_pt]))

            leg_num += 1

        if turn_radius is not None:
            turn = Turn.RIGHT if creep == 90 else Turn.LEFT
            self.waypoints = self.add_turns(
                waypoints, turn_radius, turn, delta_d=10)
        else:
            self.waypoints = [x.xy for x in waypoints]

        return self.waypoints

    def add_turns(
        self,
        waypoints: list[Waypoint],
        radius: float,
        turn: Turn,
        **kwargs,
    ) -> list[Point]:
        """Add turns to a search pattern.

        Parameters
        ----------
        waypoints: list[Waypoint]
            Search pattern waypoints.
        radius: float
            Platform turn radius, in meters.
        turn: Turn
            First turn direction.

        Returns
        -------
        list[Point]
            X- and y-coordinates of search pattern waypoints with dubins
            path turns connecting pattern legs.
        """
        new_waypoints = []

        for i, origin in enumerate(waypoints):
            new_waypoints.append(origin.xy)

            if i % 2 != 0:
                try:
                    terminus = waypoints[i+1]
                except IndexError:
                    return new_waypoints

                path = DubinsPath(origin, terminus, radius, turn)
                new_waypoints.extend(path.construct_path(**kwargs))

                turn = Turn.RIGHT if turn == Turn.LEFT else Turn.LEFT

        return new_waypoints
