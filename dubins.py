from __future__ import annotations
from enum import Enum
from itertools import pairwise
from typing import TypeAlias

import pdb # TODO remove

import numpy as np

from cartesian import calc_distance, calc_fwd
from mathlib import arccos, arctan, arctan2, cos, sin, normalize_angle
from point import Circle, Waypoint
from util import round_return


Point: TypeAlias = tuple[float, float]


class PathType(Enum):
    """Enum for Dubins path type.

    Members
    -------
    LSL: Left-Straight-Left.
    RSR: Right-Straight-Right.
    LSR: Left-Straight-Right.
    RSL: Right-Straight-Left.
    LOOPBACK: Loopback.
    """
    LSL = 1
    RSR = 2
    LSR = 3
    RSL = 4
    LOOPBACK = 5

    @classmethod
    def from_turns(cls, turns: list[Turn]) -> PathType:
        """Get the PathType from a list of Turns.

        Note: does not handle loopback case.

        Parameters
        ----------
        turns: list of Turn
            Prescribed turns.

        Returns
        -------
        PathType

        Raises
        ------
        ValueError
            If an invalid combination of turns are passed in the `turns` param.
        """
        ttype = None
        t1, t2 = turns

        if t1 == t2 == Turn.RIGHT:
            ttype = cls.RSR
        elif t1 == t2 == Turn.LEFT:
            ttype = cls.LSL
        elif t1 == Turn.RIGHT and t2 == Turn.LEFT:
            ttype = cls.RSL
        elif t1 == Turn.LEFT and t2 == Turn.RIGHT:
            ttype = cls.LSR
        else:
            raise ValueError(f'Invalid turn combination: {turns}')

        return ttype


class Turn(Enum):
    """Enum for turn direction."""
    LEFT = -1
    RIGHT = 1

    @classmethod
    def reverse(cls, turn: Turn) -> Turn:
        """Return a new Turn in the direction opposite of the given Turn."""
        if not isinstance(turn, Turn):
            raise TypeError(
                f'turn parameter must be of type Turn, got {type(turn)}')

        return Turn.RIGHT if turn == Turn.LEFT else Turn.LEFT


class DubinsBase:
    """Base class for Dubins path classes."""

    def __init__(
        self,
        origin: Waypoint,
        terminus: Waypoint,
        radius: float,
        turns: list[Turn],
    ):
        """DubinsBase constructor.

        Parameters
        ----------
        origin: Waypoint
            Fly-to Point defining the beginning of the dubins path.
        terminus: Waypoint
            Fly-to Point defining the end of the dubins path.
        radius: float
            Turn radius, in meters.
        turns: list[Turn]
            Turns to execute. Must have a length of 2.
        """
        if len(turns) != 2:
            raise ValueError(
                f'"turns" parameter must have length of 2, got {len(turns)}')

        # if turns[0] != turns[1]:
        #     raise ValueError(
        #         'Only Right-Straight-Right and Left-Straight-Left paths'
        #         ' are currently supported.')

        self.origin = origin
        self.origin.normalize()

        self.terminus = terminus
        self.terminus.normalize()

        self.radius = radius
        self.circles = None
        self.psi = None

    def _calc_circle_center(
        self,
        ftp: Waypoint,
        radius: float,
        turn: Turn,
    ) -> Circle:
        """Construct the circle for a turn.

        Parameters
        ----------
        ftp: Waypoint
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
        return Circle(
            round(ftp.x + (turn.value * radius * cos(ftp.crs)), 8),
            round(ftp.y - (turn.value * radius * sin(ftp.crs)), 8),
            turn.value)


class DubinsCSC(DubinsBase):
    """Compute a Left-Straight-Left (LSL) or Right-Straight-Right (RSR)
    Dubins path.

    Example Usage
    -------------
    >>> origin = Waypoint(0, 0, 270)
    >>> terminus = Waypoint(10, 10, 180)
    >>> radius = 3
    >>> dubins = DubinsCSC(origin, terminus, radius, [Turn.RIGHT, Turn.RIGHT])
    >>> waypoints = dubins.construct_path(delta_psi=1, delta_d=0.1)

    Reference
    ---------
    Lugo-Cárdenas, Israel & Flores, Gerardo & Salazar, Sergio & Lozano, R..
    (2014). Dubins path generation for a fixed wing UAV. 339-346.
    10.1109/ICUAS.2014.6842272.
    """

    def __init__(
        self,
        origin: Waypoint,
        terminus: Waypoint,
        radius: float,
        turns: list[Turn],
    ):
        """Instantiate a new DubinsPath.

        Parameters
        ----------
        origin: Waypoint
            Fly-to Point defining the beginning of the dubins path.
        terminus: Waypoint
            Fly-to Point defining the end of the dubins path.
        radius: float
            Turn radius, in meters.
        turns: list[Turn]
            Turns to execute. Must have a length of 2.
        """
        super().__init__(origin, terminus, radius, turns)

        self.path_type = PathType.from_turns(turns)

        self.circles = [
            self._calc_circle_center(x, radius, turn)
            for (x, turn) in zip([origin, terminus], turns)]

        self.d = self._calc_d()
        self.theta = normalize_angle(self._calc_theta())
        self.psi = origin.crs
        pdb.set_trace()

    @property
    def size(self) -> tuple[float, float]:
        """Return the size of the path in the x- and y-axis."""
        return self.d + (2 * self.radius), float(self.radius)

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
            self._calc_arc_points(self.circles[0], self.theta, delta_psi))
        waypoints.extend(self._calc_line_points(waypoints[-1], delta_d))
        waypoints.extend(
            self._calc_arc_points(
                self.circles[1], self.terminus.crs, delta_psi))

        return waypoints

    def construct_racetrack(
        self,
        delta_psi: float = 1,
        delta_d: float = 10,
    ) -> list[Point]:
        """Construct a racetrack path.

        It is up to the user to provide correct origin and terminus points.

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
        waypoints.extend(self._calc_line_points(waypoints[-1], delta_d))

        # Closure
        waypoints.append(waypoints[0])

        return waypoints

    def _calc_arc_points(
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

        psi_f = round(psi_f, 2)
        while abs(self.psi - psi_f) > delta_psi:
            psi = 90 - self.psi

            waypoints.append((
                circle.x - (circle.s * self.radius * sin(psi)),
                circle.y + (circle.s * self.radius * cos(psi)),
            ))
            self.psi = normalize_angle(self.psi + delta_psi * circle.s)

        return waypoints

    def _calc_line_points(self, origin: Point, delta: float) -> list[Point]:
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
        d_max = self.d - (delta / 2) # prevent overrun

        while d_sum < d_max:
            x_n = x_p + delta * sin(self.theta)
            y_n = y_p + delta * cos(self.theta)
            waypoints.append((x_n, y_n))

            x_p = x_n
            y_p = y_n
            d_sum += delta

        return waypoints

    @round_return(4)
    def _calc_d(self) -> float:
        """Calculate the length of the straight line segment d."""
        x_i, y_i = self.circles[0].xy
        x_f, y_f = self.circles[1].xy

        if self.path_type in [PathType.LSR, PathType.RSL]:
            lam = calc_distance(self.circles[0].xy, self.circles[1].xy)
            return np.sqrt(lam**2 - 4 * self.radius**2)

        return np.sqrt((x_f - x_i)**2 + (y_f - y_i)**2)

    @round_return(2)
    def _calc_theta(self) -> float:
        """Calculate the angle of the straight line segment d measured from
        the vertical y-axis."""
        x_i, y_i = self.circles[0].xy
        x_f, y_f = self.circles[1].xy

        if self.path_type == PathType.LSR:
            return self._calc_theta_lsr(x_i, y_i, x_f, y_f)
        elif self.path_type == PathType.RSL:
            return self._calc_theta_rsl(x_i, y_i, x_f, y_f)

        return 90 - arctan2((y_f - y_i), (x_f - x_i))

    @round_return(2)
    def _calc_theta_lsr(
        self,
        x_i: float,
        y_i: float,
        x_f: float,
        y_f: float,
    ) -> float:
        """Calculate theta for an LSR path."""
        eta = 90 + arctan2((y_f - y_i), (x_f - x_i))
        gamma = arccos((2 * self.radius) / self.d)

        return eta + gamma - 90

    @round_return(2)
    def _calc_theta_rsl(
        self,
        x_i: float,
        y_i: float,
        x_f: float,
        y_f: float,
    ) -> float:
        """Calculate theta for an RSL path."""
        eta = 90 - arctan2((y_f - y_i), (x_f - x_i))
        gamma = arccos((2 * self.radius) /self.d)

        return eta - gamma + 90



class DubinsLoopback(DubinsBase):
    """Class for constructing a loopback Dubins path.

    Example Usage
    -------------
    >>> origin = Waypoint(0, 0, 0)
    >>> terminus = Waypoint(10, 0, 180)
    >>> radius = 14
    >>> dubins = DubinsLoopback(origin, terminus, radius,
    ...                         [Turn.RIGHT, Turn.RIGHT])
    >>> waypoints = dubins.construct_path(delta_psi=1)
    """

    path_type = PathType.LOOPBACK

    def __init__(
        self,
        origin: Waypoint,
        terminus: Waypoint,
        radius: float,
        turns: list[Turn],
    ):
        """Instantiate a new DubinsPath.

        Parameters
        ----------
        origin: Waypoint
            Fly-to Point defining the beginning of the dubins path.
        terminus: Waypoint
            Fly-to Point defining the end of the dubins path.
        radius: float
            Turn radius, in meters.
        turns: list[Turn]
            Turns to execute. Must have a length of 2.
        """
        super().__init__(origin, terminus, radius, turns)

        turn1 = Turn.reverse(turns[0])
        track_spacing = self.origin.distance_to(self.terminus)
        h = np.sqrt((2 * radius)**2 - track_spacing**2)
        a = round(arccos(h / (2 * radius)), 4) + origin.crs

        if turn1 == Turn.RIGHT:
            a *= -1

        self.d = h
        self.theta = normalize_angle(a)
        self.psi = normalize_angle(origin.crs - (90. * turn1.value))

        circle1 = self._calc_circle_center(origin, radius, turn1)
        self.circles = [
            circle1,
            Circle(*calc_fwd(circle1.xy, a, self.radius*2), turns[1].value),
        ]

    @property
    def size(self) -> tuple[float, float]:
        """Return the size of the path in the x- and y-axis."""
        dist = self.origin.distance_to(self.terminus)
        s_y = self.d + self.radius

        if self.radius < dist:
            return (2 * self.radius) - dist, s_y

        return self.radius + dist, s_y

    def construct_path(self, delta_psi: float = 1) -> list[Point]:
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
            self._calc_arc_points(self.circles[0], self.theta, delta_psi))

        self.psi = normalize_angle(self.psi + 180.)

        waypoints.extend(
            self._calc_arc_points(
                self.circles[1],
                normalize_angle(self.terminus.crs - (90 * self.circles[1].s)),
                delta_psi))

        waypoints.append(calc_fwd(waypoints[-1], self.terminus.crs, self.d))

        return waypoints

    def _calc_arc_points(
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
        psi_f = round(psi_f, 2)

        #import pdb; pdb.set_trace()
        while abs(self.psi - psi_f) > delta_psi:
            waypoints.append((
                circle.x + (self.radius * sin(self.psi)),
                circle.y + (self.radius * cos(self.psi)),
            ))
            self.psi = normalize_angle(self.psi + (delta_psi * circle.s))

        return waypoints


def create_path(
    origin: Waypoint,
    terminus: Waypoint,
    radius: float,
    turns: Turn,
    **kwargs,
) -> list[Point]:
    """Create a Dubins path and return the waypoints.

    This function handles the determination of which Dubins class to use
    based on the distance between the origin and terminus points and the
    specified turn radius.

    Parameters
    ----------
    origin: Waypoint
        Fly-to Point defining the beginning of the Dubins path.
    terminus: Waypoint
        Fly-to Point defining the end of the Dubins path.
    radius: float
        Turn radius, in meters.
    turns: list[Turn]
            Turns to execute. Must have a length of 2.
    kwargs: str, optional
        Keyword arguments to pass to construct_path() methods.

    Returns
    -------
    list of tuple[float, float]
        Dubins path waypoint x- and y-coordinates.
    """
    if origin.distance_to(terminus) >= 2 * radius:
        path = DubinsCSC(origin, terminus, radius, turns)
    else:
        path = DubinsLoopback(origin, terminus, radius, turns)
        kwargs.pop('delta_d', None)

    return path.construct_path(**kwargs)


def calc_path_length(path: list[Point]) -> float:
    """Calculate the length of a Dubins path.

    Parameters
    ----------
    path: list of tuple[float, float]
        Computed Dubins path.

    Returns
    -------
    float
        Unitless length of the given path.
    """
    return round(sum([calc_distance(*x) for x in pairwise(path)]), 4)
