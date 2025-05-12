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
        if self.track < 0:
            self.track = 360 + self.track

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
    >>> delta_d = 0.1
    >>> turn = Turn.RIGHT
    >>> dubins = DubinsPath(
        origin, terminus, radius, turn, delta_psi=1, delta_d=delta_d)
    >>> waypoints = dubins.waypoints

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
        delta_psi: float = 10,
        delta_d: float = 10,
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
        turns: list[TURN]
            Turn directions.
        delta_psi: float, optional
            Interval at which to compute arc points, in degrees. Default is 10.
        delta_d: float, optional
            Interval at which to compute tangent line connecting the two
            circles, in meters. Default is 10.
        """
        self.radius = radius
        self.delta_psi = delta_psi
        self.delta_d = delta_d

        self.circles = [
            self.calc_circle_center(x, radius, turn)
            for x in [origin, terminus]]

        self.theta = self.calc_theta()
        self.d = self.calc_d()
        self.psi = origin.track
        self.waypoints = [origin.xy]
        self.compute_path(terminus.track)

    def compute_path(self, psi_f: float) -> None:
        """Construct the path.

        Parameters
        ----------
        psi_f: float
            Final heading.

        Returns
        -------
        None.
        """
        self.calc_arc_points(self.circles[0], self.theta)
        self.calc_line_points()
        self.calc_arc_points(self.circles[1], psi_f)

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

    def calc_arc_points(self, circle: Point, psi_f: float) -> None:
        """Compute the points along an arc.

        Parameters
        ----------
        circle: Circle
            Circle to rotate about.
        psi_f: float
            Final heading.

        Returns
        -------
        None.
        """
        if psi_f < self.psi:
            psi_f += 360.

        while self.psi <= psi_f:
            x_n = circle.x + (circle.s * self.radius * sin(self.psi - 90))
            y_n = circle.y + (circle.s * self.radius * cos(self.psi - 90))

            self.waypoints.append((x_n, y_n))

            self.psi += self.delta_psi

    def calc_line_points(self) -> None:
        """Compute points along the tangent line connecting the two circles.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        d_sum = 0
        x_p, y_p = self.waypoints[-1]

        while d_sum < self.d:
            x_n = x_p + self.delta_d * sin(self.theta)
            y_n = y_p + self.delta_d * cos(self.theta)

            self.waypoints.append((x_n, y_n))

            x_p = x_n
            y_p = y_n

            d_sum += self.delta_d

    @round_return(4)
    def calc_d(self) -> float:
        """Calculate the length of the straight line segment d."""
        x_i, y_i = self.circles[0].xy
        x_f, y_f = self.circles[1].xy

        return np.sqrt((x_f - x_i)**2 + (y_f - y_i)**2)

    @round_return(4)
    def calc_theta(self) -> float:
        """Calculate the angle of the straight line segment d measured from
        the vertical y-axis."""
        x_i, y_i = self.circles[0].xy
        x_f, y_f = self.circles[1].xy

        return 90 - arctan2((y_f - y_i), (x_f - x_i))

    @classmethod
    def Left(
        cls,
        origin: FTP,
        terminus: FTP,
        radius: float,
        **kwargs
    ) -> list[Point]:
        """Compute a LSL dubins path.

        Parameters
        ----------
        origin: FTP
            Fly-to Point defining the beginning of the dubins path.
        terminus: FTP
            Fly-to Point defining the end of the dubins path.
        radius: float
            Turn radius, in meters.
        kwargs: str
            Keyword arguments to pass to DubinsPath constructor.

        Returns
        -------
        list[Point]
            X- and y-coordinate pairs defining the dubins path.
        """
        dubins = DubinsPath(origin, terminus, radius, Turn.LEFT, **kwargs)

        return dubins.waypoints

    @classmethod
    def Right(
        cls,
        origin: FTP,
        terminus: FTP,
        radius: float,
        **kwargs
    ) -> list[Point]:
        """Compute a RSR dubins path.

        Parameters
        ----------
        origin: FTP
            Fly-to Point defining the beginning of the dubins path.
        terminus: FTP
            Fly-to Point defining the end of the dubins path.
        radius: float
            Turn radius, in meters.
        kwargs: str
            Keyword arguments to pass to DubinsPath constructor.

        Returns
        -------
        list[Point]
            X- and y-coordinate pairs defining the dubins path.
        """
        dubins = DubinsPath(origin, terminus, radius, Turn.RIGHT,**kwargs)

        return dubins.waypoints
