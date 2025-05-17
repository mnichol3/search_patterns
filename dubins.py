"""This module contains classes to compute dubins paths."""
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np

from mathlib import arctan2, cos, sin
from util import round_return


Point: TypeAlias = tuple[float, float]


class Turn(Enum):
    """Enum for turn direction."""
    LEFT = -1
    RIGHT = 1


@dataclass
class FTP:
    """Container for fly-to point.

    Parameters
    ----------
    x: float
        Fly-to point x-coordinate.
    y: float
        Fly-to point y-coordinate.
    track: float
        Track upon reaching the fly-to point.
    """
    x: float
    y: float
    track: float

    def __post_init__(self) -> None:
        self.track = round(normalize_angle(self.track), 2)


    @property
    def xy(self) -> tuple[float, float]:
        """Return the x- and y-coordinates."""
        return self.x, self.y


@dataclass
class Circle:
    """Container for a circle.

    Parameters
    ----------
    x: float
        X-coordinate of the center of the circle.
    y: float
        Y-coordinate of the center of the circle.
    s: int
        Direction of rotation about the circle.
        1 for clockwise, -1 for counter-clockwise.
    """
    x: float
    y: float
    s: int

    @property
    def xy(self) -> tuple[float, float]:
        """Return the x- and y-coordinates."""
        return self.x, self.y


class DubinsPath:
    """Compute a LSL or RSR dubins path.

    Example Usage
    -------------
    >>> origin = FTP(0, 0, 270)
    >>> terminus = FTP(10, 10, 180)
    >>> radius = 3
    >>> dubins = DubinsPath(origin, terminus, radius, Turn.RIGHT)
    >>> waypoints = dubins.construct_path(delta_psi=1, delta_d=0.1)

    Reference
    ---------
    Lugo-Cárdenas, Israel & Flores, Gerardo & Salazar, Sergio & Lozano, R..
    (2014). Dubins path generation for a fixed wing UAV. 339-346.
    10.1109/ICUAS.2014.6842272.
    """

    def __init__(
        self,
        origin: FTP,
        terminus: FTP,
        radius: float,
        turn: Turn,
    ):
        """Instantiate a new DubinsPath.

        Parameters
        ----------
        origin: FTP
            Fly-to Point defining the beginning of the dubins path.
        terminus: FTP
            Fly-to Point defining the end of the dubins path.
        radius: float
            Turn radius, in meters.
        turns: Turn
            Turn direction.
        """
        self.origin = origin
        self.terminus = terminus
        self.radius = radius

        self.circles = [
            self.calc_circle_center(x, radius, turn)
            for x in [origin, terminus]]

        self.theta = normalize_angle(self.calc_theta())
        self.psi = origin.track

    def construct_path(
        self,
        delta_psi: float = 1,
        delta_d: float = 10,
    ) -> list[Point]:
        """Construct a LSL or RSR path.

        Parameters
        ----------
        delta_psi: float, optional
            Interval at which to compute arc points, in degrees. Default is 10.
        delta_d: float, optional
            Interval at which to compute tangent line connecting the two
            circles, in meters. Default is 10.

        Returns
        -------
        list of Point
            X- and y-coordinates of path waypoints.
        """
        waypoints = []

        waypoints.extend(
            self.calc_arc_points(self.circles[0], self.theta, delta_psi))
        waypoints.extend(self.calc_line_points(waypoints[-1], delta_d))
        waypoints.extend(
            self.calc_arc_points(
                self.circles[1], self.terminus.track, delta_psi))

        return waypoints

    def construct_racetrack(
        self,
        delta_psi: float = 1,
        delta_d: float = 10,
    ) -> list[Point]:
        """Construct a racetrack path.

        Parameters
        ----------
        delta_psi: float, optional
            Interval at which to compute arc points, in degrees. Default is 10.
        delta_d: float, optional
            Interval at which to compute tangent line connecting the two
            circles, in meters. Default is 10.

        Returns
        -------
        list of Point
            X- and y-coordinates of path waypoints.
        """
        waypoints = self.construct_path(delta_psi=delta_psi, delta_d=delta_d)

        self.theta = normalize_angle(self.theta + 180)
        waypoints.extend(self.calc_line_points(waypoints[-1], delta_d))

        # Closure
        waypoints.append(waypoints[0])

        return waypoints

    def calc_circle_center(
        self,
        ftp: FTP,
        radius: float,
        turn: Turn,
    ) -> Circle:
        """Construct the circle for a turn.

        Parameters
        ----------
        ftp: FTP
            Fly-to point defining the beginning or end of a turn.
        radius: float
            Turn radius.
        turn: Turn
            Turn direction.

        Returns
        -------
        Circle
            Computed circle.
        """
        c_x = round(ftp.x + (turn.value * radius * cos(ftp.track)), 8)
        c_y = round(ftp.y + (-turn.value * radius * sin(ftp.track)), 8)

        return Circle(c_x, c_y, turn.value)

    def calc_arc_points(
        self,
        circle: Point,
        psi_f: float,
        delta_psi: float,
    ) -> None:
        """Compute the points along an arc.

        Parameters
        ----------
        circle: Circle
            Circle to rotate about.
        psi_f: float
            Final heading.
        delta_psi: float
            Interval at which to compute arc points, in degrees.

        Returns
        -------
        None.
        """
        waypoints = []

        # TODO really dont like this condition, easy to skip over if
        # values dont match exactly
        psi_f = round(psi_f, 2)
        while self.psi != psi_f:
            psi = 90 - self.psi

            x_n = circle.x - (circle.s * self.radius * sin(psi))
            y_n = circle.y + (circle.s * self.radius * cos(psi))

            waypoints.append((x_n, y_n))
            self.psi = self.psi + delta_psi * circle.s

            self.psi = normalize_angle(self.psi)

        return waypoints

    def calc_line_points(self, origin: Point, delta: float) -> list[Point]:
        """Compute points along the tangent line connecting the two circles.

        Parameters
        ----------
        origin: Point
            origin x- and y-coordinate.
        delta: float
            Distance delta.

        Returns
        -------
        list of Point
        """
        waypoints = []
        d_sum = 0
        x_p, y_p = origin
        d_max = self.calc_d() - (delta / 2) # prevent overrun

        while d_sum < d_max:
            x_n = x_p + delta * sin(self.theta)
            y_n = y_p + delta * cos(self.theta)

            waypoints.append((x_n, y_n))

            x_p = x_n
            y_p = y_n

            d_sum += delta

        return waypoints

    def calc_d(self) -> float:
        """Calculate the length of the straight line segment d."""
        x_i, y_i = self.circles[0].xy
        x_f, y_f = self.circles[1].xy

        return np.sqrt((x_f - x_i)**2 + (y_f - y_i)**2)

    round_return(2)
    def calc_theta(self) -> float:
        """Calculate the angle of the straight line segment d measured from
        the vertical y-axis."""
        x_i, y_i = self.circles[0].xy
        x_f, y_f = self.circles[1].xy

        return 90 - arctan2((y_f - y_i), (x_f - x_i))


def normalize_angle(val: float) -> float:
    """Normalize an angle to [-180, 180]."""
    if val > 180:
        val -= 360
    elif val < -180:
        val += 360

    return val
