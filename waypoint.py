from dataclasses import dataclass

from mathlib import normalize_angle


@dataclass
class Waypoint:
    """Contained for a waypoint consisting of an x-coordinate, y-coordinate,
    and an inbound course."""
    x: float
    y: float
    crs: float

    def __post_init__(self) -> None:
        """Post-initialization tasks."""
        self.crs %= 360.

    @property
    def xy(self) -> tuple[float, float]:
        """Return the x- and y-coordinates."""
        return self.x, self.y

    def normalize(self) -> None:
        """Normalize the course to [-180, 180]."""
        self.crs = round(normalize_angle(self.crs), 2)
