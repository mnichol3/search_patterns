"""This module constructs ISR patterns."""
from typing import TypeAlias

from cartesian import calc_fwd
from dubins import DubinsPath, FTP, Turn, normalize_angle
from mathlib import cos, sin
from search_patterns import BaseSearchPattern


Point: TypeAlias = tuple[float, float]


class Orbit(BaseSearchPattern):
    """Construct a circular orbit."""

    def __init__(self, csp: tuple[float, float]):
        """Instantiate a new Orbit object.

        Parameters
        ----------
        csp: tuple of float, float
            X- and y-coordinates of the commence search point.
        """
        super().__init__(csp)

    def run(
        self,
        inbd_crs: float,
        radius: float,
        turn_dir: int = -1,
        delta_psi: int = 1,
    ) -> list[Point]:
        """Construct an Orbit pattern.

        Parameters
        ----------
        radius: float
            Radius of the orbit, in meters.
        turn_dir: int, optional
            Turn direction. 1 = right, -1 = left. Default is -1.
        delta_psi: int
            Interval at which to compute the arch waypoints, in degrees.
            Default is 1.

        Returns
        -------
        list of tuple[float, float]
        """
        waypoints = []
        x, y = self.csp

        psi = 90 - inbd_crs
        psi_f = 90 - inbd_crs + (-turn_dir * delta_psi)

        while psi != psi_f:
            psi_adj = 90 - psi

            x_n = x - (turn_dir * radius * sin(psi_adj))
            y_n = y + (turn_dir * radius * cos(psi_adj))

            waypoints.append((x_n, y_n))
            psi = normalize_angle(psi + delta_psi * turn_dir)

        # Closure
        waypoints.append(waypoints[0])
        self.waypoints = waypoints

        return self.waypoints


class Racetrack(BaseSearchPattern):
    """Class to construct a racetrack/hold pattern."""

    def __init__(self, csp: tuple[float, float]):
        """Instantiate a new Racetrack object.

        Parameters
        ----------
        csp: tuple of float, float
            X- and y-coordinates of the commence search point.
        """
        super().__init__(csp)

    def run(
        self,
        course: int,
        d: float,
        turn_radius: float,
        turn_dir: int = -1,
        delta_psi: int = 1,
        delta_d: float = 100,
    ) -> list[Point]:
        """Construct a racetrack pattern.

        Parameters
        ----------
        course: int
            Inbound CSP course.
        d: float
            Straight segment lengths, in meters.
        turn_radius: float
            Turn radius, in meters.
        turn_dir: int, optional
            Turn direction. 1 = right, -1 = left. Default is -1.
        delta_psi: int
            Interval at which to compute the arch waypoints, in degrees.
            Default is 1.
        delta_d: float
            Interval at which to compute the straight segment waypoints,
            in meters. Default is 500.

        Returns
        -------
        list of tuple[float, float]
        """
        origin = FTP(*self.csp, course)
        terminus = FTP(
            *calc_fwd(self.csp, course + 180., d), course)

        path = DubinsPath(
            origin, terminus, turn_radius, get_turn(turn_dir))

        self.waypoints = path.construct_racetrack(
            delta_psi=delta_psi, delta_d=delta_d)

        return self.waypoints


def get_turn(turn_dir: int) -> Turn:
    """Return the Turn enum corresponding the given turn direction."""
    if turn_dir not in [-1, 1]:
        raise ValueError(f'Invalid turn_dir parameter "{turn_dir}"')

    return Turn.LEFT if turn_dir == -1 else Turn.RIGHT
