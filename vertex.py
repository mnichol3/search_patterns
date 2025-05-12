from shapely import Point


class Vertex:
    """Class to represent a vertex of the OpArea."""

    def __init__(self, x: float, y: float, name: str):
        """Instantiate a new Vertex.

        Parameters
        ----------
        x : float
            Vertex x-coordinate.
        y : float
            Vertex y-coordinate.
        name : str
            Vertex name.
        """
        self.x = x
        self.y = y
        self.name = name

    @property
    def coords(self) -> tuple[float, float]:
        """Return the x- and y-coordinates as a tuple.

        Returns
        -------
        tuple[float, float]
            Vertex x- and y-coordinates.
        """
        return (self.x, self.y)

    @property
    def point(self) -> Point:
        """Return the Vertex as a shapely.Point object.

        Returns
        -------
        Point
            Shapely.Point object containing the Vertex coordinates.
        """
        return Point(self.x, self.y)

    def __repr__(self) -> str:
        """Return a string representation of the Vertex.

        Returns
        -------
        str
            String representatino of the Vertex
        """
        return f'<Vertex {self.name} ({self.x}, {self.y})>'
