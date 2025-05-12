from typing import TypeAlias

from shapely import Geometry, LineString

from cartesian import azimuth_in_range, calc_azimuth
from vertex import Vertex
from util import round_return


CoordPair: TypeAlias = tuple[float, float]


class Edge:
    """Class to represent the edge of the OpArea."""

    def __init__(self, point1: CoordPair, point2: CoordPair, name: str):
        """Instantiate a new Edge.

        Parameters
        ----------
        point1 : CoordPair
            Vertex 1 coordinates.
        point2 : CoordPair
            Vertex 2 coordinates.
        name : str
            2-letter vertex name. Ex: "AB".
        """
        self.name = name
        self.vertices = {
            n: Vertex(*c, n) for c, n in zip([point1, point2], name)
        }
        self.linestring = LineString([point1, point2])
        self.azimuth = calc_azimuth(point1, point2)

    @property
    @round_return(2)
    def back_azimuth(self) -> float:
        """Return the inverse azimuth.

        Returns
        -------
        float
        """
        return (self.azimuth + 180.0) % 360

    @property
    def coords(self) -> tuple[CoordPair, CoordPair]:
        """Return the Edge vertice coordinates as a tuple.

        Returns
        -------
        tuple[CoordPair, CoordPair]
            Edge vertice x- and y-coordinates.
        """
        return tuple(x.coords for x in self.vertices.values())

    @property
    def length(self) -> float:
        """Return the unitless length of the Edge.

        Returns
        -------
        float
        """
        return self.linestring.length

    def intersection(self, geom: Geometry) -> Geometry:
        """Return the intersection of a shapely.Geometry and the
        Edge's LineString.

        Parameters
        ----------
        geom: shapely.Geometry

        Returns
        -------
        Geometry
        """
        return self.linestring.intersection(geom)

    def intersects(self, geom: Geometry) -> bool:
        """Return True if another shapely.Geometry intersects the Edge's
        LineString, else False.

        Parameters
        ----------
        geom: shapely.Geometry

        Returns
        -------
        bool
        """
        return self.linestring.intersects(geom)

    def __repr__(self) -> str:
        """Return a string representation of the Edge.

        Returns
        -------
        str
            String representatino of the Vertex
        """
        c1, c2 = self.coords
        return f'<Edge {self.name} [{c1}, {c2}]>'
