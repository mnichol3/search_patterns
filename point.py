from __future__ import annotations

from cartesian import calc_distance
from mathlib import normalize_angle


class Point:
    """Container for a point defined by an x- and y-coordinate."""

    def __init__(self, x: float, y: float):
        """Container for circle parameters.

        Parameters
        ----------
        x: float
            X-coordinate of the center of the point.
        y: float
            Y-coordinate of the center of the point.
        """
        self.x = x
        self.y = y

    @property
    def xy(self) -> tuple[float, float]:
        """Return the x- and y-coordinates."""
        return self.x, self.y

    def distance_to(self, p: Point) -> float:
        """Calculate the Euclidean distance from the point to another Point."""
        return calc_distance(self.xy, p.xy)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f'<{self.__class__.__name__} ({self.x}, {self.y})>'


class Circle(Point):
    """Container for circle parameters"""

    def __init__(self, x: float, y: float, s: int):
        """Instantiate a new Circle.

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
        if s not in [-1, 1]:
            raise ValueError(f'"s" parameter must be in [-1, 1], got {s}')

        super().__init__(x, y)
        self.s = s

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f'<{self.__class__.__name__} ({self.x}, {self.y}), s={self.s}>'


class Waypoint(Point):
    """Container for a waypoint consisting of an x-coordinate, y-coordinate,
    and an inbound course."""

    def __init__(self, x: float, y: float, crs: float):
        """Instantiate a new Waypoint.

        Parameters
        ----------
        x: float
            X-coordinate of the center of the waypoint.
        y: float
            Y-coordinate of the center of the waypoint.
        crs: int
            Inbound course.
        """
        super().__init__(x, y)
        self.crs = crs % 360.

    def normalize(self) -> None:
        """Normalize the course to [-180, 180]."""
        self.crs = round(normalize_angle(self.crs), 2)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return (f'<{self.__class__.__name__} ({self.x}, {self.y}),'
                f' crs={self.crs}>')
