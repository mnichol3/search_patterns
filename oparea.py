from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from itertools import pairwise
from string import ascii_uppercase
from typing import TypeAlias, TYPE_CHECKING

import numpy as np
from shapely import Geometry, Point, Polygon, STRtree

from cartesian import calc_azimuth, get_min_azimuth_diff
from dubins import DubinsPath
from edge import Edge
from mathlib import cos, sin, NMI_2_M
from search_patterns import ExpandingSquare, ParallelTrackSearch, SectorSearch
from tmtransformer import TMTransformer
from util import get_coord_mean
from vertex import Vertex
from utm_zone import UTMZone

try:
    import simplekml
except ImportError:
    pass

if TYPE_CHECKING:
    from pathlib import Path


CoordPair: TypeAlias = tuple[float, float]


class BaseOpArea(ABC):
    """OpArea base class."""

    def __init__(
        self,
        coords: list[CoordPair],
        major_axis: float,
        transform: bool = False,
    ):
        coords.append(coords[0])

        if transform:
            coords = self.transform_fwd(coords)

        self.polygon = Polygon(coords)
        self.edges = self._init_edges(coords)
        self.major_axis = major_axis
        self.patterns = {}

    @property
    def area(self) -> float:
        """Return the area of the OpArea, in meters squared."""
        return self.polygon.area

    @property
    @abstractmethod
    def convergence(self) -> float:
        """Return the convergence angle."""
        pass

    @property
    def coords(self) -> list[CoordPair]:
        """Return the coordinates of the OpArea vertices as longitude
        & latitude pairs."""
        return self.transform_inv([x.coords for x in self.vertices.values()])

    @property
    def vertices(self) -> dict[str, Vertex]:
        """Return the OpArea vertices as a dict of Vertex objects.

        Parameters
        ----------
        None.

        Returns
        -------
        dict of str, Vertex
        """
        vertices = {}
        for e in self.edges.values():
            for k, v in e.vertices.items():
                if k not in vertices:
                    vertices[k] = v

        return vertices

    def get_nearest_vertex(
        self,
        curr_pt: tuple[float, float] | Point,
    ) -> tuple[Vertex, float]:
        """Return the vertex closest to the given point and the distance
        between the two.

        Parameters
        ----------
        curr_pt: tuple[float, float] or shapely.Point
            Point of interest.

        Returns
        -------
        Vertex
            Vertex closest to the given point.
        float
            Unitless distance between the given point and the closest vertex.
        """
        if not isinstance(curr_pt, Point):
            curr_pt = Point(*curr_pt)

        tree, items = self._get_vertex_tree()
        idx, dist = tree.query_nearest(curr_pt, return_distance=True)

        vert = self.vertices[items.take(idx).tolist()[0]]
        dist = round(float(dist[0]), 2)

        return vert, dist

    def generate_expanding_square_search(
        self,
        csp: tuple[float, float],
        first_course: int,
        d: float,
        unit_id: str,
        num_legs: int = 12,
        turn_dir: int = -1,
    ) -> any:
        """Generate the sector search pattern.

        Parameters
        ----------
        csp: tuple[float, float]
            Longitude and latitude coordinates of the commence search point.
        first_course: int
            Course of the first leg of the pattern.
        d: float
            Initial leg length, in nautical miles.
        unit_id: str
            Name of the unit assigned to the pattern.
        num_legs: int, optional
            Number of legs. Default is 12.
        turn_dir: int, optional
            Turn direction. 1 for right, -1 for left. Default is -1.

        Returns
        -------
        list[tuple[float, float]]
            Pattern waypoints.
        """
        d *= NMI_2_M

        search = ExpandingSquare(self.transform_fwd(csp))
        search.run(first_course, d, num_legs=num_legs, turn_dir=turn_dir)
        self.patterns[unit_id] = search.to_dataframe(self.transform_inv)

    def generate_parallel_track_search(
        self,
        csp: tuple[float, float],
        creep: float,
        track_spacing: float,
        unit_id: str,
        axis: str = 'major',
        first_course: float | None = None,
        turn_radius: float | None = None,
    ) -> any:
        """Construct a parallel track search pattern.

        Parameters
        ----------
        csp: tuple[float, float]
            Longitude and latitude coordinates of the commence search point.
        creep: float
            Relative creep direction (either 90 or -90).
        track_spacing: float
            Track spacing, in nautical miles.
        unit_id: str
            Name of the unit assigned to the pattern.

        Returns
        -------
        any
            _description_
        """

        def _get_leg_course() -> float:
            edges = sorted(
                [x for x in self.edges.values() if start_vert.name in x.name],
                key=lambda e: e.length)

            if axis == 'major':
                edge = edges[-1]
            else:
                edge = edges[0]

            centroid_azimuth = calc_azimuth(
                start_vert.coords,
                (self.polygon.centroid.x, self.polygon.centroid.y))

            d_azi = get_min_azimuth_diff(centroid_azimuth, edge.azimuth)
            d_b_azi = get_min_azimuth_diff(centroid_azimuth, edge.back_azimuth)
            # d_azi = get_min_azimuth_diff(self.major_axis, edge.azimuth)
            # d_b_azi = get_min_azimuth_diff(self.major_axis, edge.back_azimuth)

            if d_azi < d_b_azi:
                return edge.azimuth
            else:
                return edge.back_azimuth

        track_spacing *= NMI_2_M

        start_vert, _ = self.get_nearest_vertex(self.transform_fwd(csp))

        search = ParallelTrackSearch(self.polygon, start_vert.coords)

        if first_course is None:
            first_course = _get_leg_course()

        search.run(
            first_course, (first_course + creep) % 360., track_spacing,
            turn_radius=turn_radius)

        self.patterns[unit_id] = search.to_dataframe(self.transform_inv)

    def generate_sector_search(
        self,
        csp: tuple[float, float],
        orientation: int,
        radius: int,
        unit_id: str,
        n_patterns: int = 1,
    ) -> any:
        """Generate the sector search pattern.

        Parameters
        ----------
        csp: tuple[float, float]
            Longitude and latitude coordinates of the commence search point.
        orientation: int
            Pattern orientation, in degrees.
        radius: int
            Pattern radius, in nautical miles.
        unit_id: str
            ID of the unit to conduct the search.
        n_patterns : int, optional
            Number of patterns to generate, by default 1.

        Returns
        -------
        list[tuple[float, float]]
            Pattern waypoints.
        """
        proj_csp = self.transform_fwd(csp)

        if not self.contains_properly(Point(*proj_csp)):
            raise ValueError('Commence search point is outside OpArea.')

        search = SectorSearch(proj_csp)
        search.run(orientation, radius*NMI_2_M, n_patterns)
        self.patterns[unit_id] = search.to_dataframe(self.transform_inv)

    def to_kml(
        self,
        path: str | Path,
        name: str,
        fill: str | None = None,
        doc_name: str | None = None,
    ) -> None:
        """Write the OpArea to a KML as a polygon.

        Parameters
        ----------
        path: str or pathlib.Path
            Path of the KML file to write.
        name: str
            Name of the KML polygon.
        fill: str, optional
            Polygon fill color.

        Returns
        -------
        None.
        """
        if 'simplekml' not in sys.modules:
            raise ImportError('simplekml module not loaded.')

        kml = simplekml.Kml()

        if doc_name:
            kml.document.name = doc_name

        bbox = kml.newpolygon(name=name, outerboundaryis=self.coords)

        if fill:
            bbox.style.polystyle.color = getattr(simplekml.Color, fill)

        for pattern, df in self.patterns.items():
           ls = kml.newlinestring(
               name=pattern, coords=list(zip(df['lon'], df['lat'])))
           ls.style.linestyle.color = simplekml.Color.red
           ls.style.linestyle.width = 3

        kml.save(path)

    @abstractmethod
    def transform_fwd(self, coords: list[CoordPair]) -> list[CoordPair]:
        """Perform a forward coordinate system transform.

        Parameters
        ----------
        coords: list of tuple[float, float]

        Returns
        -------
        list of tuple[float, float]
        """
        pass

    @abstractmethod
    def transform_inv(self, coords: list[CoordPair]) -> list[CoordPair]:
        """Perform a inverse coordinate system transform.

        Parameters
        ----------
        coords: list of tuple[float, float]

        Returns
        -------
        list of tuple[float, float]
        """
        pass

    def contains(self, geom: Geometry) -> bool:
        """Return True if another shapely.Geometry is entirely inside the
        OpArea's polygon, else False.

        Parameters
        ----------
        geom: shapely.Geometry

        Returns
        -------
        bool
        """
        return self.polygon.contains(geom)

    def contains_properly(self, geom: Geometry) -> bool:
        """Return True if another shapely.Geometry is entirely inside the
        OpArea's polygon with no common boundary points, else False.

        Parameters
        ----------
        geom: shapely.Geometry

        Returns
        -------
        bool
        """
        return self.polygon.contains_properly(geom)

    def intersection(self, geom: Geometry) -> Geometry:
        """Return the intersection of a shapely.Geometry and the
        OpArea's polygon.

        Parameters
        ----------
        geom: shapely.Geometry

        Returns
        -------
        Geometry
        """
        return self.polygon.intersection(geom)

    def intersects(self, geom: Geometry) -> bool:
        """Return True if another shapely.Geometry intersects the OpArea's
        polygon, else False.

        Parameters
        ----------
        geom: shapely.Geometry

        Returns
        -------
        bool
        """
        return self.polygon.intersects(geom)

    @classmethod
    def from_datum(
        cls,
        centroid: CoordPair,
        length: float,
        width: float,
        major_axis_azimuth: float,
    ) -> BaseOpArea:
        """Generate an OpArea from a centroid, length, width, and axis azimuth.

        Parameters
        ----------
        centroid: tuple[float, float]
            Longitude and latitude coordinates of the bounding box centroid.
        length: float
            Length of the OpArea, in nautical miles.
            This is the longer length compared to width.
        width: float
            Width of the OpArea, in nautical miles.
            This is the shorter distance compared to length.
        major_axis_azimuth: float
            Azimuth of the OpArea major axis. This will be the azimuth of the
            longest edges of the OpArea.

        Returns
        -------
        OpArea
        """
        pass

    def _init_edges(self, coords) -> dict[str, Edge]:
        """Initialize the OpArea Edges.

        Parameters
        ----------
        coords: list of tuple of float, float
            List of x- and y-coordinates.

        Returns
        -------
        dict of str, Edge
        """
        edges = {}
        v_names = [ascii_uppercase[i] for i in range(len(coords)-1)]
        v_names.append(v_names[0])

        for n_tpl, c_pair in zip(pairwise(v_names), pairwise(coords)):
            name = ''.join(n_tpl)
            edges[name] = Edge(*c_pair, name)

        return edges

    def _get_major_axis(self) -> float:
        """Return the azimuth of the OpArea's major axis."""
        max_len = -1
        axis_azimuth = 0

        for e in self.edges.values():
            if e.length > max_len:
                max_len = e.length
                axis_azimuth = e.azimuth

        return axis_azimuth

    def _get_vertex_tree(self) -> tuple[STRtree, np.array]:
        names = []
        geoms = []

        for k, v in self.vertices.items():
            names.append(k)
            geoms.append(v.point)

        return STRtree(geoms), np.array(names)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return (
            f'<{self.__class__.__name__}'
            f' {[tuple(round(x, 4) for x in c) for c in self.coords]}>')


class TMOpArea(BaseOpArea):
    """An OpArea subclass using a custom transverse mercator projection."""

    def __init__(
        self,
        coords: list[CoordPair],
        major_axis: float,
        transformer: TMTransformer | None = None,
        transform: bool = False,
    ):
        """Instantiate a new OpArea.

        Parameters
        ----------
        coords: list of CoordPair
            List of x- and y-coordinate pairs defining the extent of the
            OpArea.
        utm_zone: UTMZone
            UTM zone used to project the coordinates to UTM.
        """
        self.transformer = transformer

        if transformer is None:
            self.transformer = TMTransformer.from_lonlat(
                *get_coord_mean(coords))

        super().__init__(coords, major_axis, transform)

    @property
    def convergence(self) -> float:
        """Return the convergence angle."""
        return 0.0

    def transform_fwd(self, coords: list[CoordPair]) -> list[CoordPair]:
        """Perform a forward coordinate system transform.

        Parameters
        ----------
        coords: list of tuple[float, float]

        Returns
        -------
        list of tuple[float, float]
        """
        return self.transformer.fwd(coords)

    def transform_inv(self, coords: list[CoordPair]) -> list[CoordPair]:
        """Perform a inverse coordinate system transform.

        Parameters
        ----------
        coords: list of tuple[float, float]

        Returns
        -------
        list of tuple[float, float]
        """
        return self.transformer.inv(coords)

    @classmethod
    def from_datum(
        cls,
        centroid: CoordPair,
        length: float,
        width: float,
        major_axis_azimuth: float,
    ) -> UTMOpArea:
        """Generate an OpArea from a centroid, length, width, and axis azimuth.

        Parameters
        ----------
        centroid: tuple[float, float]
            Longitude and latitude coordinates of the bounding box centroid.
        length: float
            Length of the OpArea, in nautical miles.
            This is the longer length compared to width.
        width: float
            Width of the OpArea, in nautical miles.
            This is the shorter distance compared to length.
        major_axis_azimuth: float
            Azimuth of the OpArea major axis. This will be the azimuth of the
            longest edges of the OpArea.

        Returns
        -------
        OpArea
        """
        vertices = []
        length *= NMI_2_M
        width *= NMI_2_M

        transformer = TMTransformer.from_lonlat(*centroid)
        cx, cy = transformer.fwd(centroid)

        dx = length / 2
        dy = width / 2

        major_axis_azimuth = 90 - major_axis_azimuth

        ux = cos(major_axis_azimuth)
        uy = sin(major_axis_azimuth)
        vx = -sin(major_axis_azimuth)
        vy = cos(major_axis_azimuth)

        vertices = [
            (cx + dx * ux + dy * vx, cy + dx * uy + dy * vy),
            (cx - dx * ux + dy * vx, cy - dx * uy + dy * vy),
            (cx - dx * ux - dy * vx, cy - dx * uy - dy * vy),
            (cx + dx * ux - dy * vx, cy + dx * uy - dy * vy),
        ]

        return TMOpArea(vertices, major_axis_azimuth, transformer=transformer)


class UTMOpArea(BaseOpArea):
    """An OpArea subclass using universal transverse mercator projection."""

    def __init__(
        self,
        coords: list[CoordPair],
        major_axis: float,
        utm_zone: UTMZone,
        transform: bool = False,
    ):
        """Instantiate a new OpArea.

        Parameters
        ----------
        coords: list of CoordPair
            List of x- and y-coordinate pairs defining the extent of the
            OpArea.
        utm_zone: UTMZone
            UTM zone used to project the coordinates to UTM.
        """
        self.utm_zone = utm_zone
        super().__init__(coords, major_axis, transform)

    @property
    def convergence(self) -> float:
        """Return the convergence angle."""
        self.utm_zone.convergence

    def transform_fwd(self, coords: list[CoordPair]) -> list[CoordPair]:
        """Perform a forward coordinate system transform.

        Parameters
        ----------
        coords: list of tuple[float, float]

        Returns
        -------
        list of tuple[float, float]
        """
        return self.utm_zone.geodetic_to_utm(coords)

    def transform_inv(self, coords: list[CoordPair]) -> list[CoordPair]:
        """Perform a inverse coordinate system transform.

        Parameters
        ----------
        coords: list of tuple[float, float]

        Returns
        -------
        list of tuple[float, float]
        """
        return self.utm_zone.utm_to_geodetic(coords)

    @classmethod
    def from_datum(
        cls,
        centroid: CoordPair,
        length: float,
        width: float,
        major_axis_azimuth: float,
    ) -> UTMOpArea:
        """Generate an OpArea from a centroid, length, width, and axis azimuth.

        Parameters
        ----------
        centroid: tuple[float, float]
            Longitude and latitude coordinates of the bounding box centroid.
        length: float
            Length of the OpArea, in nautical miles.
            This is the longer length compared to width.
        width: float
            Width of the OpArea, in nautical miles.
            This is the shorter distance compared to length.
        major_axis_azimuth: float
            Azimuth of the OpArea major axis. This will be the azimuth of the
            longest edges of the OpArea.

        Returns
        -------
        OpArea
        """
        vertices = []
        length *= NMI_2_M
        width *= NMI_2_M

        utm_zone = UTMZone.from_lonlat(centroid)
        cx, cy = utm_zone.geodetic_to_utm(centroid)

        dx = length / 2
        dy = width / 2

        major_axis_azimuth = 90 - major_axis_azimuth + utm_zone.convergence

        ux = cos(major_axis_azimuth)
        uy = sin(major_axis_azimuth)
        vx = -sin(major_axis_azimuth)
        vy = cos(major_axis_azimuth)

        vertices = [
            (cx + dx * ux + dy * vx, cy + dx * uy + dy * vy),
            (cx - dx * ux + dy * vx, cy - dx * uy + dy * vy),
            (cx - dx * ux - dy * vx, cy - dx * uy - dy * vy),
            (cx + dx * ux - dy * vx, cy + dx * uy - dy * vy),
        ]

        return UTMOpArea(vertices, major_axis_azimuth, utm_zone)
