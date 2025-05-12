"""Tests for Edge class."""
from edge import Edge


coords = [(83.75, 0.0), (83.75, 120.5)]


def test_init() -> None:
    """Test basic instantiation.

    Returns
    -------
    None.

    Raises
    ------
    AssertionError
    """
    t_edge = Edge(*coords, 'AB')

    assert t_edge.name == 'AB'
    assert t_edge.coords[0] == (83.75, 0.0)
    assert t_edge.coords[1] == (83.75, 120.5)


def test_vertices() -> None:
    """Test Vertex instantiation.

    Returns
    -------
    None.

    Raises
    ------
    AssertionError
    """
    t_edge = Edge(*coords, 'AB')

    assert list(t_edge.vertices.keys()) == ['A', 'B']

    assert t_edge.vertices['A'].x == 83.75
    assert t_edge.vertices['A'].y == 0.0

    assert t_edge.vertices['B'].x == 83.75
    assert t_edge.vertices['B'].y == 120.5
