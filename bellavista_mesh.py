#!/usr/bin/env python3
"""
bellavista_mesh.py

Read a GeoJSON/JSON file containing polygon vertices for Bellavista,
project the longitude/latitude coordinates to a local Cartesian coordinate
system in meters, generate a 2D mesh with Gmsh, and plot the mesh with
Matplotlib.

The local Cartesian coordinate system is chosen so that

    (x, y) = (0, 0)

corresponds to the user-specified reference point

    latitude  = -0.6958902314708109 degrees
    longitude = -90.32546837031445 degrees

which is equivalent to

    0.6958902314708109 S, 90.32546837031445 W.

Dependencies
------------
- numpy
- pyproj
- gmsh
- matplotlib

Example
-------
    python bellavista_mesh.py bellavista.json --h 10 --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from pyproj import CRS, Transformer

try:
    import gmsh
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "This script requires the 'gmsh' Python package. Install it with `pip install gmsh`."
    ) from exc


# -----------------------------------------------------------------------------
# Fixed reference point supplied by the user.
# Negative latitude means south; negative longitude means west.
# -----------------------------------------------------------------------------
ORIGIN_LAT = -0.6958902314708109
ORIGIN_LON = -90.32546837031445


def _load_coordinates_from_geojson(json_path: str | Path) -> np.ndarray:
    """
    Load longitude/latitude coordinates from a GeoJSON-like JSON file.

    This helper is intentionally tolerant of a few common structures:
    - FeatureCollection -> first feature -> geometry -> coordinates
    - Feature -> geometry -> coordinates
    - raw geometry dict with a "coordinates" field

    Supported geometry types
    ------------------------
    - LineString: coordinates = [[lon, lat], [lon, lat], ...]
    - Polygon:    coordinates = [[[lon, lat], [lon, lat], ...], ...]
                  In this case, the exterior ring is used.

    Parameters
    ----------
    json_path : str or Path
        Path to the input JSON/GeoJSON file.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing [lon, lat] pairs.

    Raises
    ------
    ValueError
        If the file does not contain a supported geometry.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Resolve down to a geometry object.
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError("FeatureCollection contains no features.")
        geometry = features[0].get("geometry", {})
    elif data.get("type") == "Feature":
        geometry = data.get("geometry", {})
    else:
        geometry = data

    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    if coords is None:
        raise ValueError("No 'coordinates' field found in the JSON geometry.")

    if gtype == "LineString":
        arr = np.asarray(coords, dtype=float)
    elif gtype == "Polygon":
        if len(coords) == 0:
            raise ValueError("Polygon geometry contains no rings.")
        arr = np.asarray(coords[0], dtype=float)  # exterior ring only
    else:
        raise ValueError(
            f"Unsupported geometry type: {gtype!r}. Expected 'LineString' or 'Polygon'."
        )

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Coordinates must be an array of [longitude, latitude] pairs.")

    # If the input repeats the first point at the end, remove the duplicate.
    # Gmsh only needs each geometric vertex once; the closing segment will be
    # added explicitly later.
    if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]

    if len(arr) < 3:
        raise ValueError("At least three distinct vertices are required to define a polygon.")

    return arr


def latlon_to_cartesian(
    json_path: str | Path,
    origin_lat: float = ORIGIN_LAT,
    origin_lon: float = ORIGIN_LON,
) -> np.ndarray:
    """
    Convert polygon vertices from longitude/latitude to local Cartesian meters.

    The conversion is performed using `pyproj` with a local azimuthal
    equidistant projection centered at the given origin. This has two useful
    properties for the present task:

    1. The chosen origin maps exactly to (0, 0).
    2. Distances from the origin are represented in meters.

    Parameters
    ----------
    json_path : str or Path
        Path to the GeoJSON/JSON file containing the polygon vertices.
    origin_lat : float, optional
        Latitude of the local origin in decimal degrees.
    origin_lon : float, optional
        Longitude of the local origin in decimal degrees.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) whose rows are [x, y] coordinates in meters.
        Here x points approximately east and y points approximately north.

    Notes
    -----
    - The input JSON is assumed to store coordinates in GeoJSON order:
      [longitude, latitude].
    - The output array preserves the vertex ordering from the file.
    """
    lonlat = _load_coordinates_from_geojson(json_path)
    lon = lonlat[:, 0]
    lat = lonlat[:, 1]

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),  # WGS84 geographic coordinates
        local_crs,
        always_xy=True,
    )

    x, y = transformer.transform(lon, lat)
    return np.column_stack((x, y))


def build_polygon_mesh(
    vertices: np.ndarray,
    h: float = 10.0,
    refinement_parameters: tuple[float, float] = (50.0, 1.0),
    mesh_algorithm: int | None = None,
    order: int = 1,
    recombine: bool = False,
    optimize: bool = False,
    element_size_min: float | None = None,
    element_size_max: float | None = None,
    model_name: str = "polygon_mesh",
    verbose: bool = False,
    finalize: bool = True,
) -> dict[str, Any]:
    """
    Build a 2D Gmsh mesh for a polygon, with optional local refinement near
    the origin.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (N, 2) containing polygon vertices [x, y] in meters.
        The vertices are assumed to be ordered around the boundary.
    h : float, optional
        Baseline characteristic mesh size away from the release point. The
        default value h=10 aims for triangles of side length on the order of
        10 meters over most of the domain.
    refinement_parameters : tuple(float, float), optional
        Tuple ``(refinement_radius, max_refinement_degree)`` controlling local
        mesh refinement around the embedded point (0, 0).

        - ``refinement_radius`` is the radius, in meters, of a disk centered at
          the origin inside which the mesh is refined.
        - ``max_refinement_degree`` controls the smallest target element size
          near the origin. The local target size is taken to be

              h_min = h / 10**max_refinement_degree.

          For the default ``(50, 1)``, this gives ``h_min = 1`` meter when
          ``h = 10``. The mesh then transitions smoothly from about 1 meter near
          the origin to about 10 meters outside the 50 meter refinement disk.

        The point (0, 0) is explicitly embedded in the mesh whether or not it is
        one of the polygon boundary vertices.
    mesh_algorithm : int or None, optional
        Optional Gmsh 2D meshing algorithm number. If None, the Gmsh default is
        used.
    order : int, optional
        Polynomial order of mesh elements. The default is 1 (linear elements).
    recombine : bool, optional
        If True, request recombination of triangles into quadrilaterals where
        possible.
    optimize : bool, optional
        If True, run Gmsh mesh optimization after generation.
    element_size_min : float or None, optional
        Optional global minimum mesh size. If supplied, this acts as an overall
        lower bound in addition to the local refinement field.
    element_size_max : float or None, optional
        Optional global maximum mesh size.
    model_name : str, optional
        Name assigned to the Gmsh model.
    verbose : bool, optional
        If False, suppress most Gmsh terminal output.
    finalize : bool, optional
        If True (default), finalize the Gmsh session before returning.

    Returns
    -------
    dict
        Dictionary containing mesh data extracted from Gmsh, including:

        - "vertices": original input vertices
        - "node_coords": array of node coordinates, shape (num_nodes, 2)
        - "triangles": triangle connectivity, shape (num_triangles, 3)
        - "lines": boundary edge connectivity, shape (num_edges, 2)
        - "gmsh_node_tags": original Gmsh node tags
        - "gmsh_element_types": list of returned Gmsh element types
        - "origin_node_index": zero-based index of the mesh node at (or
          numerically closest to) the embedded origin point

    Notes
    -----
    The refinement is implemented using a Gmsh background mesh field based on
    distance from the embedded origin point. This is a natural way to obtain a
    graded mesh that is very fine near the release point and gradually becomes
    coarser away from it.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("'vertices' must have shape (N, 2).")
    if len(vertices) < 3:
        raise ValueError("At least three vertices are required.")
    if h <= 0:
        raise ValueError("Mesh size 'h' must be positive.")
    if order < 1:
        raise ValueError("Element order must be at least 1.")

    try:
        refinement_radius, max_refinement_degree = refinement_parameters
    except Exception as exc:
        raise ValueError(
            "'refinement_parameters' must be a tuple like (radius, degree)."
        ) from exc

    refinement_radius = float(refinement_radius)
    max_refinement_degree = float(max_refinement_degree)

    if refinement_radius <= 0:
        raise ValueError("Refinement radius must be positive.")
    if max_refinement_degree < 0:
        raise ValueError("Maximum refinement degree must be nonnegative.")

    # The smallest target element size near the origin.  With the default
    # h=10 and degree=1, this is 1 meter.
    local_h_min = h / (10.0 ** max_refinement_degree)
    if element_size_min is not None:
        local_h_min = max(float(element_size_min), local_h_min)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add(model_name)

        if mesh_algorithm is not None:
            gmsh.option.setNumber("Mesh.Algorithm", mesh_algorithm)
        if element_size_min is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", float(element_size_min))
        if element_size_max is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(element_size_max))

        # When a background field is used, disabling size extension from points
        # and curvature helps the field control the grading more predictably.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        # ------------------------------------------------------------------
        # 1. Create one Gmsh point for each polygon vertex.
        # ------------------------------------------------------------------
        point_tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, h) for x, y in vertices]

        # ------------------------------------------------------------------
        # 2. Create and embed the origin point (0, 0) into the interior of the
        #    polygon.  This ensures that the point source location is present as
        #    a mesh node, which is very convenient for later PDE forcing.
        # ------------------------------------------------------------------
        origin_tag = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, local_h_min)

        # ------------------------------------------------------------------
        # 3. Connect consecutive boundary points with line segments and form the
        #    polygonal surface.
        # ------------------------------------------------------------------
        line_tags = []
        n = len(point_tags)
        for i in range(n):
            start_tag = point_tags[i]
            end_tag = point_tags[(i + 1) % n]
            line_tags.append(gmsh.model.geo.addLine(start_tag, end_tag))

        curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        if recombine:
            gmsh.model.geo.mesh.setRecombine(2, surface)

        # Synchronize the CAD kernel before using mesh-level operations like
        # embedded points and background fields.
        gmsh.model.geo.synchronize()

        # Embed the origin point into the 2D surface so that it appears as an
        # actual mesh node even though it is not part of the boundary.
        gmsh.model.mesh.embed(0, [origin_tag], 2, surface)

        # ------------------------------------------------------------------
        # 4. Build a graded background field controlling element sizes as a
        #    function of distance from the embedded origin point.
        #
        #    The Distance field measures distance to the origin point.  The
        #    Threshold field then requests size local_h_min near the origin and
        #    size h away from the origin, interpolating smoothly in between.
        # ------------------------------------------------------------------
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "PointsList", [origin_tag])

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", local_h_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", h)

        # Use a short inner plateau of very fine elements near the source, then
        # gradually relax to the background size by the requested radius.
        dist_min = 0.25 * refinement_radius
        dist_max = refinement_radius
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", dist_max)

        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

        # Set element order and generate the 2D mesh.
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.generate(2)

        if optimize:
            gmsh.model.mesh.optimize()

        # ------------------------------------------------------------------
        # Extract node coordinates.
        # Gmsh returns node coordinates as a flat array [x1,y1,z1,x2,y2,z2,...].
        # ------------------------------------------------------------------
        node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
        node_xyz = np.asarray(node_coords_flat, dtype=float).reshape(-1, 3)
        node_xy = node_xyz[:, :2]

        # Build a map from Gmsh node tag -> zero-based row index in node_xy.
        tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}

        # Identify the embedded origin node by nearest distance.  In a correct
        # embedded mesh this should land exactly on (0, 0) up to roundoff.
        origin_node_index = int(np.argmin(np.linalg.norm(node_xy, axis=1)))

        # ------------------------------------------------------------------
        # Extract element connectivity.
        # We separate line elements (boundary) and triangle elements (interior).
        # For higher-order elements, we keep only the corner nodes when plotting.
        # ------------------------------------------------------------------
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
        triangles: list[list[int]] = []

        for etype, conn in zip(elem_types, elem_node_tags):
            props = gmsh.model.mesh.getElementProperties(etype)
            element_name = props[0]
            num_nodes_per_elem = props[3]
            conn = np.asarray(conn, dtype=int).reshape(-1, num_nodes_per_elem)

            if "Triangle" in element_name:
                triangles.extend(
                    [[tag_to_index[int(row[0])], tag_to_index[int(row[1])], tag_to_index[int(row[2])]] for row in conn]
                )

        boundary_types, _, boundary_conn = gmsh.model.mesh.getElements(1)
        lines: list[list[int]] = []

        for etype, conn in zip(boundary_types, boundary_conn):
            props = gmsh.model.mesh.getElementProperties(etype)
            element_name = props[0]
            num_nodes_per_elem = props[3]
            conn = np.asarray(conn, dtype=int).reshape(-1, num_nodes_per_elem)

            if "Line" in element_name:
                lines.extend(
                    [[tag_to_index[int(row[0])], tag_to_index[int(row[-1])]] for row in conn]
                )

        mesh_data = {
            "vertices": vertices.copy(),
            "node_coords": node_xy,
            "triangles": np.asarray(triangles, dtype=int) if triangles else np.empty((0, 3), dtype=int),
            "lines": np.asarray(lines, dtype=int) if lines else np.empty((0, 2), dtype=int),
            "gmsh_node_tags": np.asarray(node_tags, dtype=int),
            "gmsh_element_types": list(elem_types),
            "origin_node_index": origin_node_index,
        }

        return mesh_data
    finally:
        if finalize:
            gmsh.finalize()


def plot_mesh(
    mesh_data: dict[str, Any],
    ax: plt.Axes | None = None,
    show_nodes: bool = False,
    show_boundary: bool = True,
    boundary_linewidth: float = 1.0,
    triangle_linewidth: float = 0.4,
    node_size: float = 8.0,
    equal_aspect: bool = True,
    title: str = "Bellavista mesh",
    xlabel: str = "x (m)",
    ylabel: str = "y (m)",
    show: bool = True,
) -> plt.Axes:
    """
    Plot the mesh geometry using Matplotlib.

    Parameters
    ----------
    mesh_data : dict
        Dictionary returned by `build_polygon_mesh`.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes on which to draw. If None, a new figure/axes pair is
        created.
    show_nodes : bool, optional
        If True, draw mesh nodes as points.
    show_boundary : bool, optional
        If True, emphasize the polygon boundary edges.
    boundary_linewidth : float, optional
        Line width used when drawing the boundary.
    triangle_linewidth : float, optional
        Line width used for the triangular mesh edges.
    node_size : float, optional
        Marker size for nodes if `show_nodes=True`.
    equal_aspect : bool, optional
        If True, enforce equal axis scaling so the geometry is not distorted.
    title, xlabel, ylabel : str, optional
        Plot labeling options.
    show : bool, optional
        If True, call `plt.show()` before returning.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    node_xy = np.asarray(mesh_data["node_coords"], dtype=float)
    triangles = np.asarray(mesh_data.get("triangles", []), dtype=int)
    lines = np.asarray(mesh_data.get("lines", []), dtype=int)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Draw the triangular mesh if present.
    if len(triangles) > 0:
        ax.triplot(node_xy[:, 0], node_xy[:, 1], triangles, linewidth=triangle_linewidth)

    # Draw boundary segments with a slightly heavier style so the outer polygon
    # is easy to identify visually.
    if show_boundary and len(lines) > 0:
        segments = np.stack((node_xy[lines[:, 0]], node_xy[lines[:, 1]]), axis=1)
        lc = LineCollection(segments, linewidths=boundary_linewidth)
        ax.add_collection(lc)

    if show_nodes:
        ax.scatter(node_xy[:, 0], node_xy[:, 1], s=node_size)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.autoscale()
    ax.margins(0.03)

    if show:
        plt.show()

    return ax

def get_mesh_area(mesh_data):
    triangles = mesh_data['triangles']
    coords = mesh_data['node_coords']

    p0 = coords[triangles[:, 0]]
    p1 = coords[triangles[:, 1]]
    p2 = coords[triangles[:, 2]]

    v1 = p0 - p2
    v2 = p1 - p2

    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.sum(areas)


def main() -> None:
    """
    Run the full Bellavista workflow from the command line.

    Steps carried out
    -----------------
    1. Read the input JSON/GeoJSON file.
    2. Convert longitude/latitude vertices to local Cartesian meters.
    3. Build a 2D Gmsh mesh.
    4. Print a short summary to the terminal.
    5. Optionally display a Matplotlib plot of the mesh.
    """
    parser = argparse.ArgumentParser(
        description="Project Bellavista polygon vertices to meters, mesh the polygon with Gmsh, and plot the result."
    )
    parser.add_argument(
        "json_file",
        type=str,
        nargs="?",
        default="bellavista.json",
        help="Path to the Bellavista JSON/GeoJSON file (default: bellavista.json).",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=10.0,
        help="Characteristic mesh size in meters (default: 10).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Mesh element order (default: 1).",
    )
    parser.add_argument(
        "--refinement-radius",
        type=float,
        default=50.0,
        help="Radius in meters of the refinement disk around (0, 0) (default: 50).",
    )
    parser.add_argument(
        "--refinement-degree",
        type=float,
        default=1.0,
        help="Maximum local refinement degree near (0, 0); local h is h/10**degree (default: 1).",
    )
    parser.add_argument(
        "--mesh-algorithm",
        type=int,
        default=None,
        help="Optional Gmsh 2D mesh algorithm number.",
    )
    parser.add_argument(
        "--recombine",
        action="store_true",
        help="Request recombination into quadrilateral elements where possible.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Gmsh mesh optimization after generation.",
    )
    parser.add_argument(
        "--show-nodes",
        action="store_true",
        help="Show mesh nodes in the Matplotlib plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the Matplotlib plot.",
    )
    args = parser.parse_args()

    vertices_xy = latlon_to_cartesian(args.json_file)
    mesh_data = build_polygon_mesh(
        vertices_xy,
        h=args.h,
        refinement_parameters=(args.refinement_radius, args.refinement_degree),
        mesh_algorithm=args.mesh_algorithm,
        order=args.order,
        recombine=args.recombine,
        optimize=args.optimize,
    )
    
    total_area = get_mesh_area(mesh_data)

    print("Projected polygon vertices (meters):")
    print(vertices_xy)
    print()
    print(f"Number of boundary vertices : {len(vertices_xy)}")
    print(f"Number of mesh nodes        : {len(mesh_data['node_coords'])}")
    print(f"Number of triangles         : {len(mesh_data['triangles'])}")
    print(f"Number of boundary segments : {len(mesh_data['lines'])}")
    print(f"Origin node index           : {mesh_data['origin_node_index']}")
    print(f"Origin node coordinates     : {mesh_data['node_coords'][mesh_data['origin_node_index']]}")
    print(f"Total mesh area             : {total_area} m^2")

    plot_mesh(mesh_data, show_nodes=args.show_nodes, show=True)
    
    return mesh_data


if __name__ == "__main__":
    mesh_data = main()
