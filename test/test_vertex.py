"""Tests for Vertex class."""
from shapely import Point

from vertex import Vertex


coord = (83.75, 120.5)
name = 'C'


def test_init() -> None:
    """Test basic instantiation.

    Returns
    -------
    None.

    Raises
    ------
    AssertionError
    """
    v = Vertex(*coord, name)

    assert v.x == 83.75
    assert v.y == 120.5
    assert v.name == 'C'
    assert v.coords == coord


def test_as_point() -> None:
    """Assert Vertex.as_point() returns a shapely.LineString.

    Returns
    -------
    None.

    Raises
    ------
    AssertionError
    """
    v = Vertex(*coord, name)
    v_point = v.as_point()

    assert isinstance(v_point, Point)
    assert v_point.x == 83.75
    assert v_point.y == 120.5
