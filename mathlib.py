import numpy as np
import numpy.typing as npt


NMI_2_M = 1852
M_2_NMI = round(1 / NMI_2_M, 8)


def arcsin(vals: float | list[float]) -> float | npt.NDArray[np.float64]:
    """Compute the trigonometric inverse sine and return the value
    in degrees."""
    return np.degrees(np.arcsin(vals))


def arctan(vals: float | list[float]) -> float | npt.NDArray[np.float64]:
    """Compute the trigonometric inverse tangent and return the value
    in degrees."""
    return np.degrees(np.arctan(vals))


def arctan2(
    y: float | list[float],
    x: float | list[float],
) -> float | npt.NDArray[np.float64]:
    """Compute the trigonometric inverse tangent and return the value
    in degrees."""
    return np.degrees(np.arctan2(y, x))


def cos(vals: float | list[float]) -> float | npt.NDArray[np.float64]:
    """Compute the cosine of the given values."""
    return np.cos(np.radians(vals))


def sin(vals: float | list[float]) -> float | npt.NDArray[np.float64]:
    """Compute the sin of the given values."""
    return np.sin(np.radians(vals))


def sec(vals: float | list[float]) -> float | npt.NDArray[np.float64]:
    """Compute the secant of the given values."""
    return 1 / np.cos(np.radians(vals))


def sech(vals: float | list[float]) -> float | npt.NDArray[np.float64]:
  """Compute the hyperbolic secant of the given values."""
  return 1 / np.cosh(vals)


def tan(vals: float | list[float]) -> float | npt.NDArray[np.float64]:
    """Compute the sin of the given values."""
    return np.tan(np.radians(vals))
