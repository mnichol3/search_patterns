"""Tests for OpArea class."""
import pytest

from oparea import OpArea


@pytest.fixture
def oparea_coords() -> list[tuple[float, float]]:
    """Coordinates to define an OpArea for testing.

    Returns
    -------
    list of tuple of float, float
        OpArea x- and y-coordinates.
    """
    return [
        (23.75, 0.0),
        (83.75, 0.0),
        (83.75, 120.5),
        (23.75, 120.5),
    ]


def test_init(oparea_coords: list[tuple[float, float]]) -> None:
    """Test basic instantiation.

    Returns
    -------
    None.

    Raises
    ------
    AssertionError
    """
    oparea = OpArea(oparea_coords)

    assert list(oparea.edges.keys()) == ['AB', 'BC', 'CD', 'DA']
    assert list(oparea.vertices.keys()) == ['A', 'B', 'C', 'D']
