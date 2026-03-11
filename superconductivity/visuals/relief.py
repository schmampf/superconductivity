"""Visible relief extraction for regular height fields.

This module turns a regular scalar field ``z[y_i, x_j]`` into a set of
view-dependent visible outline polylines. The terrain is treated as a
triangulated top surface, projected from an explicit camera, and reduced to
the subset of silhouette segments that remain visible after hidden-line
removal.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from superconductivity.utilities.types import LIM, NDArray64

from .helper import check_xyz, get_xylim_indices

try:  # pragma: no cover - depends on local environment.
    from numba import njit
except ModuleNotFoundError:  # pragma: no cover - exercised via fallback only.

    def njit(*args, **kwargs):
        """Fallback no-op decorator when Numba is unavailable."""

        def decorator(func):
            return func

        return decorator


def _as_point3(value: NDArray64 | tuple[float, float, float]) -> NDArray64:
    """Convert a 3-vector-like input to a float64 NumPy array."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("Expected a 3-vector with shape (3,).")
    return arr


def _normalize(vec: NDArray64, *, name: str) -> NDArray64:
    """Return a unit-length copy of ``vec``."""
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return vec / norm


def _wrap_progress(
    iterable: Iterable[Any],
    *,
    total: int,
    progress: bool,
    desc: str,
) -> Iterable[Any]:
    """Wrap an iterable in ``tqdm`` when progress reporting is requested."""
    if not progress:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env.
        raise ImportError(
            "progress=True requires the optional 'tqdm' package.",
        ) from exc

    return tqdm(iterable, total=total, desc=desc)


def _progress_bar(
    *,
    total: int,
    progress: bool,
    desc: str,
):
    """Create an optional ``tqdm`` progress bar context manager."""
    if not progress:
        return None

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env.
        raise ImportError(
            "progress=True requires the optional 'tqdm' package.",
        ) from exc

    return tqdm(total=total, desc=desc)


def _normalize_progress_mode(progress_mode: str) -> str:
    """Validate and normalize the requested progress update mode."""
    mode = str(progress_mode).strip().lower()
    if mode not in {"auto", "chunks", "candidates"}:
        raise ValueError(
            "progress_mode must be 'auto', 'chunks', or 'candidates'.",
        )
    return mode


@dataclass(frozen=True)
class CameraSpec:
    """Camera definition for relief extraction.

    Parameters
    ----------
    observer
        Camera position in world coordinates ``(x, y, z)``.
    target
        Point the camera looks at in world coordinates.
    up
        Approximate world up-direction used to build the camera basis.
    projection
        Projection mode. Supported values are ``"perspective"`` and
        ``"orthographic"``.
    """

    observer: NDArray64 | tuple[float, float, float]
    target: NDArray64 | tuple[float, float, float]
    up: NDArray64 | tuple[float, float, float] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    projection: str = "perspective"

    def __post_init__(self) -> None:
        """Validate and normalize camera fields."""
        observer = _as_point3(self.observer)
        target = _as_point3(self.target)
        up = _as_point3(self.up)
        projection = str(self.projection).strip().lower()

        if projection not in {"perspective", "orthographic"}:
            raise ValueError(
                "projection must be 'perspective' or 'orthographic'.",
            )
        if np.allclose(observer, target):
            raise ValueError("observer and target must differ.")

        object.__setattr__(self, "observer", observer)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "projection", projection)


@dataclass(frozen=True)
class VisibleRelief:
    """Visible outline geometry extracted from a height field.

    Parameters
    ----------
    polylines
        List of visible projected polylines. Each polyline has shape
        ``(N, 2)`` in camera screen coordinates.
    world_segments
        Optional list of visible source segments in world coordinates. Each
        segment has shape ``(2, 3)``.
    camera
        Camera specification used during extraction.
    screen_bounds
        Bounds of the projected terrain as
        ``((xmin, xmax), (ymin, ymax))``.
    """

    polylines: list[NDArray64]
    world_segments: Optional[list[NDArray64]]
    camera: CameraSpec
    screen_bounds: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class PreparedReliefMesh:
    """Prepared terrain mesh for repeated relief extraction."""

    x: NDArray64
    y: NDArray64
    z: NDArray64
    vertices: NDArray64
    triangles: np.ndarray
    edges: np.ndarray
    edge_faces: np.ndarray
    tri_normals: NDArray64
    tri_centroids: NDArray64


@dataclass(frozen=True)
class _CameraFrame:
    """Derived camera basis and projection data."""

    spec: CameraSpec
    right: NDArray64
    up: NDArray64
    forward: NDArray64


@dataclass(frozen=True)
class _UniformGrid:
    """Uniform 2D spatial index over projected screen-space boxes."""

    bounds_min: NDArray64
    bounds_max: NDArray64
    nx: int
    ny: int
    cell_starts: np.ndarray
    cell_items: np.ndarray
    n_items: int


@dataclass(frozen=True)
class _ProjectionState:
    """Camera-dependent projected mesh state used during visibility tests."""

    edge_vertex_ids: np.ndarray
    edge_world: NDArray64
    edge_camera: NDArray64
    edge_projected: NDArray64
    edge_box_min: NDArray64
    edge_box_max: NDArray64
    tri_projected: NDArray64
    tri_camera_z: NDArray64
    tri_box_min: NDArray64
    tri_box_max: NDArray64
    edge_index: _UniformGrid
    tri_index: _UniformGrid
    candidate_edges: np.ndarray
    perspective: bool
    split_tol: float
    query_tol: float
    ray_eps: float
    screen_bounds: tuple[tuple[float, float], tuple[float, float]]


_WORKER_STATE: _ProjectionState | None = None


def _build_camera_frame(camera: CameraSpec) -> _CameraFrame:
    """Construct an orthonormal camera frame from a camera specification."""
    forward = _normalize(camera.target - camera.observer, name="target vector")
    right_raw = np.cross(forward, camera.up)
    right_norm = float(np.linalg.norm(right_raw))
    if right_norm <= 0.0:
        raise ValueError("up vector must not be parallel to the view direction.")
    right = right_raw / right_norm
    up = _normalize(np.cross(right, forward), name="camera up")

    return _CameraFrame(
        spec=camera,
        right=right,
        up=up,
        forward=forward,
    )


def _project_vertices(
    vertices: NDArray64,
    camera: _CameraFrame,
) -> tuple[NDArray64, NDArray64]:
    """Project world-space vertices into camera coordinates and screen space."""
    rel = vertices - camera.spec.observer
    basis = np.column_stack((camera.right, camera.up, camera.forward))
    camera_coords = rel @ basis
    depths = camera_coords[:, 2]

    if camera.spec.projection == "perspective":
        near_eps = 1e-9
        if np.any(depths <= near_eps):
            raise ValueError(
                "Perspective projection requires all selected terrain "
                "points to lie in front of the observer.",
            )
        projected = camera_coords[:, :2] / depths[:, None]
    else:
        projected = camera_coords[:, :2]

    return camera_coords.astype(np.float64), projected.astype(np.float64)


def _build_mesh(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
) -> tuple[NDArray64, np.ndarray]:
    """Build vertices and upward-oriented triangles from a regular grid."""
    xg, yg = np.meshgrid(x, y, indexing="xy")
    vertices = np.column_stack((xg.ravel(), yg.ravel(), z.ravel()))

    ny, nx = z.shape
    triangles: list[tuple[int, int, int]] = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            v00 = iy * nx + ix
            v01 = v00 + 1
            v10 = (iy + 1) * nx + ix
            v11 = v10 + 1

            triangles.append((v00, v01, v11))
            triangles.append((v00, v11, v10))

    return vertices.astype(np.float64), np.asarray(triangles, dtype=np.int64)


def _build_edge_table(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build unique mesh edges and their adjacent face indices."""
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_idx, tri in enumerate(triangles):
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(key, []).append(face_idx)

    edges: list[tuple[int, int]] = []
    edge_faces: list[tuple[int, int]] = []
    for edge, faces in edge_to_faces.items():
        edges.append(edge)
        if len(faces) == 1:
            edge_faces.append((faces[0], -1))
        elif len(faces) == 2:
            edge_faces.append((faces[0], faces[1]))

    return (
        np.asarray(edges, dtype=np.int64),
        np.asarray(edge_faces, dtype=np.int64),
    )


def prepare_relief_mesh(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
    *,
    xlim: LIM = None,
    ylim: LIM = None,
) -> PreparedReliefMesh:
    """Prepare a triangulated relief mesh for repeated camera queries.

    Parameters
    ----------
    x
        1D x-axis values of shape ``(Nx,)``.
    y
        1D y-axis values of shape ``(Ny,)``.
    z
        2D height field of shape ``(Ny, Nx)`` with ``z[y_i, x_j]``.
    xlim
        Optional x-range used to crop the field before meshing.
    ylim
        Optional y-range used to crop the field before meshing.

    Returns
    -------
    PreparedReliefMesh
        Reusable mesh data for camera-dependent relief extraction.
    """
    x_arr, y_arr, z_arr = check_xyz(x=x, y=y, z=z)
    ix, iy = get_xylim_indices(x=x_arr, y=y_arr, xlim=xlim, ylim=ylim)
    x_sel = np.asarray(x_arr[ix], dtype=np.float64)
    y_sel = np.asarray(y_arr[iy], dtype=np.float64)
    z_sel = np.asarray(z_arr[np.ix_(iy, ix)], dtype=np.float64)

    vertices, triangles = _build_mesh(x=x_sel, y=y_sel, z=z_sel)
    edges, edge_faces = _build_edge_table(triangles=triangles)

    tri_v0 = vertices[triangles[:, 0]]
    tri_v1 = vertices[triangles[:, 1]]
    tri_v2 = vertices[triangles[:, 2]]
    tri_e1 = tri_v1 - tri_v0
    tri_e2 = tri_v2 - tri_v0
    tri_normals = np.cross(tri_e1, tri_e2)
    tri_centroids = tri_v0 + (tri_e1 + tri_e2) / 3.0

    return PreparedReliefMesh(
        x=x_sel,
        y=y_sel,
        z=z_sel,
        vertices=vertices,
        triangles=triangles,
        edges=edges,
        edge_faces=edge_faces,
        tri_normals=tri_normals.astype(np.float64),
        tri_centroids=tri_centroids.astype(np.float64),
    )


def _face_visibility(
    mesh: PreparedReliefMesh,
    camera: _CameraFrame,
) -> NDArray64:
    """Compute signed face visibility values."""
    if camera.spec.projection == "perspective":
        view = camera.spec.observer[None, :] - mesh.tri_centroids
    else:
        view = np.broadcast_to(-camera.forward, mesh.tri_normals.shape)

    return np.einsum("ij,ij->i", mesh.tri_normals, view)


def _classify_sign(values: NDArray64, eps: float) -> np.ndarray:
    """Classify signed values into ``{-1, 0, +1}`` with tolerance."""
    out = np.zeros(values.shape, dtype=np.int8)
    out[values > eps] = 1
    out[values < -eps] = -1
    return out


def _collect_candidate_edges(
    mesh: PreparedReliefMesh,
    face_values: NDArray64,
) -> np.ndarray:
    """Collect silhouette candidate edges from precomputed adjacency."""
    face_eps = 1e-12 * max(1.0, float(np.max(np.abs(face_values))))
    face_sign = _classify_sign(face_values, eps=face_eps)

    candidates: list[tuple[int, int, int]] = []
    for edge_idx, (u, v) in enumerate(mesh.edges):
        f0, f1 = map(int, mesh.edge_faces[edge_idx])
        if f1 < 0:
            if face_sign[f0] >= 0:
                candidates.append((edge_idx, int(u), int(v)))
            continue

        s0 = int(face_sign[f0])
        s1 = int(face_sign[f1])
        if s0 == s1 and s0 != 0:
            continue
        if s0 == 0 and s1 == 0:
            continue
        candidates.append((edge_idx, int(u), int(v)))

    return np.asarray(candidates, dtype=np.int64)


@njit(cache=True)
def _segment_intersection_parameter_value(
    p0: NDArray64,
    p1: NDArray64,
    q0: NDArray64,
    q1: NDArray64,
    tol: float,
) -> float:
    """Return a segment-intersection parameter or ``-1`` when absent."""
    r0 = p1[0] - p0[0]
    r1 = p1[1] - p0[1]
    s0 = q1[0] - q0[0]
    s1 = q1[1] - q0[1]
    qp0 = q0[0] - p0[0]
    qp1 = q0[1] - p0[1]
    denom = r0 * s1 - r1 * s0

    if abs(denom) <= tol:
        return -1.0

    t = (qp0 * s1 - qp1 * s0) / denom
    u = (qp0 * r1 - qp1 * r0) / denom
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return t
    return -1.0


def _segment_intersection_parameter(
    p0: NDArray64,
    p1: NDArray64,
    q0: NDArray64,
    q1: NDArray64,
    tol: float,
) -> Optional[float]:
    """Return the parameter of a proper 2D segment intersection on ``p``."""
    t = float(
        _segment_intersection_parameter_value(
            p0=p0,
            p1=p1,
            q0=q0,
            q1=q1,
            tol=tol,
        )
    )
    if t < 0.0:
        return None
    return t


def _unique_parameters(
    params: list[float],
    tol: float,
) -> list[float]:
    """Sort and merge nearly identical edge split parameters."""
    params_sorted = sorted(float(np.clip(p, 0.0, 1.0)) for p in params)
    unique: list[float] = []
    for value in params_sorted:
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(value)
        else:
            unique[-1] = 0.5 * (unique[-1] + value)

    if not unique:
        return [0.0, 1.0]

    unique[0] = 0.0
    unique[-1] = 1.0
    return unique


@njit(cache=True)
def _edge_parameter_from_screen_value(
    screen_point: NDArray64,
    edge_cam0: NDArray64,
    edge_cam1: NDArray64,
    perspective: bool,
    eps: float,
) -> float:
    """Return the 3D edge parameter matching a screen-space point."""
    if perspective:
        dx = edge_cam1[0] - edge_cam0[0]
        dy = edge_cam1[1] - edge_cam0[1]
        dz = edge_cam1[2] - edge_cam0[2]

        den_x = screen_point[0] * dz - dx
        den_y = screen_point[1] * dz - dy

        if abs(den_x) >= abs(den_y):
            if abs(den_x) <= eps:
                return np.nan
            return (edge_cam0[0] - screen_point[0] * edge_cam0[2]) / den_x

        if abs(den_y) <= eps:
            return np.nan
        return (edge_cam0[1] - screen_point[1] * edge_cam0[2]) / den_y

    den_x = edge_cam1[0] - edge_cam0[0]
    den_y = edge_cam1[1] - edge_cam0[1]
    if abs(den_x) >= abs(den_y):
        if abs(den_x) <= eps:
            return np.nan
        return (screen_point[0] - edge_cam0[0]) / den_x

    if abs(den_y) <= eps:
        return np.nan
    return (screen_point[1] - edge_cam0[1]) / den_y


def _edge_parameter_from_screen_point(
    screen_point: NDArray64,
    edge_cam0: NDArray64,
    edge_cam1: NDArray64,
    *,
    perspective: bool,
    eps: float,
) -> Optional[float]:
    """Return the 3D edge parameter matching a screen-space point."""
    t = float(
        _edge_parameter_from_screen_value(
            screen_point=screen_point,
            edge_cam0=edge_cam0,
            edge_cam1=edge_cam1,
            perspective=perspective,
            eps=eps,
        )
    )
    if not np.isfinite(t):
        return None
    return t


@njit(cache=True)
def _triangle_depth_value(
    screen_point: NDArray64,
    tri_screen: NDArray64,
    tri_depths: NDArray64,
    perspective: bool,
    eps: float,
) -> float:
    """Return triangle depth at a screen point, or ``inf`` if outside."""
    ax = tri_screen[0, 0]
    ay = tri_screen[0, 1]
    bx = tri_screen[1, 0]
    by = tri_screen[1, 1]
    cx = tri_screen[2, 0]
    cy = tri_screen[2, 1]
    px = screen_point[0]
    py = screen_point[1]

    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) <= eps:
        return np.inf

    l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
    l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
    l2 = 1.0 - l0 - l1

    tol = 8.0 * eps
    if l0 < -tol or l1 < -tol or l2 < -tol:
        return np.inf

    if perspective:
        inv_depth = (
            l0 / tri_depths[0]
            + l1 / tri_depths[1]
            + l2 / tri_depths[2]
        )
        if inv_depth <= eps:
            return np.inf
        return 1.0 / inv_depth

    return l0 * tri_depths[0] + l1 * tri_depths[1] + l2 * tri_depths[2]


def _choose_grid_shape(
    n_items: int,
    span: NDArray64,
) -> tuple[int, int]:
    """Choose a modest uniform grid shape from item count and aspect ratio."""
    if n_items <= 0:
        return 1, 1

    span_safe = np.maximum(span, 1e-12)
    aspect = float(span_safe[0] / span_safe[1])
    approx_cells = max(1, int(np.ceil(n_items / 8.0)))
    nx = max(1, int(np.sqrt(approx_cells * aspect)))
    ny = max(1, int(np.ceil(approx_cells / nx)))
    return min(nx, 128), min(ny, 128)


@njit(cache=True)
def _grid_ranges_raw(
    bounds_min: NDArray64,
    bounds_max: NDArray64,
    nx: int,
    ny: int,
    box_min: NDArray64,
    box_max: NDArray64,
) -> tuple[int, int, int, int]:
    """Map a screen-space box to integer grid ranges."""
    span0 = max(bounds_max[0] - bounds_min[0], 1e-12)
    span1 = max(bounds_max[1] - bounds_min[1], 1e-12)

    fx0 = (box_min[0] - bounds_min[0]) / span0
    fx1 = (box_max[0] - bounds_min[0]) / span0
    fy0 = (box_min[1] - bounds_min[1]) / span1
    fy1 = (box_max[1] - bounds_min[1]) / span1

    ix0 = int(np.floor(nx * fx0))
    ix1 = int(np.floor(nx * fx1))
    iy0 = int(np.floor(ny * fy0))
    iy1 = int(np.floor(ny * fy1))

    ix0 = min(max(ix0, 0), nx - 1)
    ix1 = min(max(ix1, 0), nx - 1)
    iy0 = min(max(iy0, 0), ny - 1)
    iy1 = min(max(iy1, 0), ny - 1)

    return ix0, ix1, iy0, iy1


def _grid_ranges(
    index: _UniformGrid,
    box_min: NDArray64,
    box_max: NDArray64,
) -> tuple[int, int, int, int]:
    """Map a screen-space box to integer grid ranges."""
    return _grid_ranges_raw(
        bounds_min=index.bounds_min,
        bounds_max=index.bounds_max,
        nx=index.nx,
        ny=index.ny,
        box_min=box_min,
        box_max=box_max,
    )


def _build_uniform_grid(
    box_min: NDArray64,
    box_max: NDArray64,
    bounds_min: NDArray64,
    bounds_max: NDArray64,
) -> _UniformGrid:
    """Build a compact uniform grid over screen-space bounding boxes."""
    nx, ny = _choose_grid_shape(
        n_items=int(box_min.shape[0]),
        span=np.maximum(bounds_max - bounds_min, 1e-12),
    )
    ncells = nx * ny
    counts = np.zeros(ncells, dtype=np.int64)

    for item_idx in range(int(box_min.shape[0])):
        ix0, ix1, iy0, iy1 = _grid_ranges_raw(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            nx=nx,
            ny=ny,
            box_min=box_min[item_idx],
            box_max=box_max[item_idx],
        )
        for iy in range(iy0, iy1 + 1):
            row = iy * nx
            for ix in range(ix0, ix1 + 1):
                counts[row + ix] += 1

    cell_starts = np.zeros(ncells + 1, dtype=np.int64)
    np.cumsum(counts, out=cell_starts[1:])
    cell_items = np.empty(int(cell_starts[-1]), dtype=np.int64)
    offsets = cell_starts[:-1].copy()

    for item_idx in range(int(box_min.shape[0])):
        ix0, ix1, iy0, iy1 = _grid_ranges_raw(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            nx=nx,
            ny=ny,
            box_min=box_min[item_idx],
            box_max=box_max[item_idx],
        )
        for iy in range(iy0, iy1 + 1):
            row = iy * nx
            for ix in range(ix0, ix1 + 1):
                cell_id = row + ix
                insert_at = offsets[cell_id]
                cell_items[insert_at] = item_idx
                offsets[cell_id] += 1

    return _UniformGrid(
        bounds_min=bounds_min.astype(np.float64),
        bounds_max=bounds_max.astype(np.float64),
        nx=nx,
        ny=ny,
        cell_starts=cell_starts,
        cell_items=cell_items,
        n_items=int(box_min.shape[0]),
    )


def _query_uniform_grid(
    index: _UniformGrid,
    box_min: NDArray64,
    box_max: NDArray64,
    marks: np.ndarray,
    stamp: int,
) -> tuple[np.ndarray, int]:
    """Query deduplicated item ids overlapping a screen-space box."""
    if stamp >= np.iinfo(np.int64).max - 1:
        marks.fill(0)
        stamp = 1

    ix0, ix1, iy0, iy1 = _grid_ranges(index=index, box_min=box_min, box_max=box_max)
    out: list[int] = []
    for iy in range(iy0, iy1 + 1):
        row = iy * index.nx
        for ix in range(ix0, ix1 + 1):
            cell_id = row + ix
            start = int(index.cell_starts[cell_id])
            stop = int(index.cell_starts[cell_id + 1])
            for pos in range(start, stop):
                item_idx = int(index.cell_items[pos])
                if marks[item_idx] != stamp:
                    marks[item_idx] = stamp
                    out.append(item_idx)

    return np.asarray(out, dtype=np.int64), stamp + 1


def _merge_segments(
    segments: list[NDArray64],
    tol: float,
) -> list[NDArray64]:
    """Merge connected visible segments into ordered projected polylines."""
    if not segments:
        return []

    def make_key(point: NDArray64) -> tuple[int, int]:
        scaled = np.rint(point / tol).astype(np.int64)
        return int(scaled[0]), int(scaled[1])

    node_index: dict[tuple[int, int], int] = {}
    node_points: list[list[NDArray64]] = []
    adjacency: list[list[tuple[int, int]]] = []
    edges: list[tuple[int, int]] = []

    for seg in segments:
        if seg.shape != (2, 2):
            raise ValueError("Projected segments must have shape (2, 2).")
        a = make_key(seg[0])
        b = make_key(seg[1])
        for key, point in ((a, seg[0]), (b, seg[1])):
            if key not in node_index:
                node_index[key] = len(node_points)
                node_points.append([point])
                adjacency.append([])
            else:
                node_points[node_index[key]].append(point)

        ia = node_index[a]
        ib = node_index[b]
        edge_idx = len(edges)
        edges.append((ia, ib))
        adjacency[ia].append((edge_idx, ib))
        adjacency[ib].append((edge_idx, ia))

    node_coords = np.asarray(
        [np.mean(points, axis=0) for points in node_points],
        dtype=np.float64,
    )

    used = np.zeros(len(edges), dtype=bool)
    polylines: list[NDArray64] = []

    def walk(start_edge: int, start_node: int) -> NDArray64:
        coords = [node_coords[start_node]]
        edge_idx = start_edge
        current = start_node

        while True:
            used[edge_idx] = True
            node_a, node_b = edges[edge_idx]
            next_node = node_b if current == node_a else node_a
            coords.append(node_coords[next_node])

            candidates = [
                (cand_edge, cand_node)
                for cand_edge, cand_node in adjacency[next_node]
                if not used[cand_edge]
            ]
            if len(candidates) != 1:
                break

            edge_idx, current = candidates[0][0], next_node

        return np.asarray(coords, dtype=np.float64)

    for node_id, neighbors in enumerate(adjacency):
        if len(neighbors) == 2:
            continue
        for edge_idx, _ in neighbors:
            if used[edge_idx]:
                continue
            polylines.append(walk(edge_idx, node_id))

    for edge_idx, (node_a, _) in enumerate(edges):
        if used[edge_idx]:
            continue
        polylines.append(walk(edge_idx, node_a))

    return polylines


def _build_projection_state(
    mesh: PreparedReliefMesh,
    frame: _CameraFrame,
) -> _ProjectionState:
    """Build camera-dependent projected mesh state."""
    camera_vertices, projected_vertices = _project_vertices(
        vertices=mesh.vertices,
        camera=frame,
    )

    edge_projected = projected_vertices[mesh.edges]
    edge_camera = camera_vertices[mesh.edges]
    edge_world = mesh.vertices[mesh.edges]
    edge_box_min = np.min(edge_projected, axis=1)
    edge_box_max = np.max(edge_projected, axis=1)

    tri_projected = projected_vertices[mesh.triangles]
    tri_camera_z = camera_vertices[mesh.triangles][:, :, 2]
    tri_box_min = np.min(tri_projected, axis=1)
    tri_box_max = np.max(tri_projected, axis=1)

    screen_min = np.min(projected_vertices, axis=0)
    screen_max = np.max(projected_vertices, axis=0)
    screen_span = np.maximum(screen_max - screen_min, 1.0)
    split_tol = 1e-9 * float(np.max(screen_span))
    query_tol = 4.0 * split_tol

    face_values = _face_visibility(mesh=mesh, camera=frame)
    candidate_edges = _collect_candidate_edges(mesh=mesh, face_values=face_values)

    edge_index = _build_uniform_grid(
        box_min=edge_box_min,
        box_max=edge_box_max,
        bounds_min=screen_min,
        bounds_max=screen_max,
    )
    tri_index = _build_uniform_grid(
        box_min=tri_box_min,
        box_max=tri_box_max,
        bounds_min=screen_min,
        bounds_max=screen_max,
    )

    return _ProjectionState(
        edge_vertex_ids=mesh.edges.astype(np.int64),
        edge_world=edge_world.astype(np.float64),
        edge_camera=edge_camera.astype(np.float64),
        edge_projected=edge_projected.astype(np.float64),
        edge_box_min=edge_box_min.astype(np.float64),
        edge_box_max=edge_box_max.astype(np.float64),
        tri_projected=tri_projected.astype(np.float64),
        tri_camera_z=tri_camera_z.astype(np.float64),
        tri_box_min=tri_box_min.astype(np.float64),
        tri_box_max=tri_box_max.astype(np.float64),
        edge_index=edge_index,
        tri_index=tri_index,
        candidate_edges=candidate_edges.astype(np.int64),
        perspective=frame.spec.projection == "perspective",
        split_tol=split_tol,
        query_tol=query_tol,
        ray_eps=1e-10,
        screen_bounds=(
            (float(screen_min[0]), float(screen_max[0])),
            (float(screen_min[1]), float(screen_max[1])),
        ),
    )


def _extract_from_state_range(
    state: _ProjectionState,
    candidate_indices: np.ndarray,
) -> tuple[list[NDArray64], list[NDArray64]]:
    """Extract visible segments for a subset of candidate edges."""
    visible_segments_2d: list[NDArray64] = []
    visible_segments_3d: list[NDArray64] = []
    edge_marks = np.zeros(state.edge_index.n_items, dtype=np.int64)
    tri_marks = np.zeros(state.tri_index.n_items, dtype=np.int64)
    edge_stamp = 1
    tri_stamp = 1

    for cand_idx in candidate_indices:
        edge_idx = int(state.candidate_edges[cand_idx, 0])
        p0 = state.edge_projected[edge_idx, 0]
        p1 = state.edge_projected[edge_idx, 1]
        edge_world0 = state.edge_world[edge_idx, 0]
        edge_world1 = state.edge_world[edge_idx, 1]
        edge_cam0 = state.edge_camera[edge_idx, 0]
        edge_cam1 = state.edge_camera[edge_idx, 1]

        seg_min = np.minimum(p0, p1) - state.query_tol
        seg_max = np.maximum(p0, p1) + state.query_tol
        overlap_edges, edge_stamp = _query_uniform_grid(
            index=state.edge_index,
            box_min=seg_min,
            box_max=seg_max,
            marks=edge_marks,
            stamp=edge_stamp,
        )
        if overlap_edges.size > 0:
            overlap_edges = overlap_edges[
                np.all(state.edge_box_max[overlap_edges] >= seg_min, axis=1)
                & np.all(state.edge_box_min[overlap_edges] <= seg_max, axis=1)
            ]

        params = [0.0, 1.0]
        for other_edge_idx in overlap_edges:
            q0 = state.edge_projected[other_edge_idx, 0]
            q1 = state.edge_projected[other_edge_idx, 1]
            t_hit = _segment_intersection_parameter(
                p0=p0,
                p1=p1,
                q0=q0,
                q1=q1,
                tol=state.split_tol,
            )
            if t_hit is not None:
                params.append(t_hit)

        params_unique = _unique_parameters(params=params, tol=state.split_tol)
        for left, right in zip(params_unique[:-1], params_unique[1:]):
            if right - left <= state.split_tol:
                continue

            mid_param = 0.5 * (left + right)
            screen_mid = p0 + mid_param * (p1 - p0)

            edge_mid_t = _edge_parameter_from_screen_point(
                screen_point=screen_mid,
                edge_cam0=edge_cam0,
                edge_cam1=edge_cam1,
                perspective=state.perspective,
                eps=state.ray_eps,
            )
            if edge_mid_t is None:
                continue

            edge_depth = edge_cam0[2] + edge_mid_t * (edge_cam1[2] - edge_cam0[2])
            tri_ids, tri_stamp = _query_uniform_grid(
                index=state.tri_index,
                box_min=screen_mid - state.query_tol,
                box_max=screen_mid + state.query_tol,
                marks=tri_marks,
                stamp=tri_stamp,
            )
            if tri_ids.size > 0:
                tri_ids = tri_ids[
                    np.all(
                        state.tri_box_max[tri_ids] >= screen_mid - state.query_tol,
                        axis=1,
                    )
                    & np.all(
                        state.tri_box_min[tri_ids] <= screen_mid + state.query_tol,
                        axis=1,
                    )
                ]

            nearest_depth = np.inf
            for tri_id in tri_ids:
                depth = float(
                    _triangle_depth_value(
                        screen_point=screen_mid,
                        tri_screen=state.tri_projected[tri_id],
                        tri_depths=state.tri_camera_z[tri_id],
                        perspective=state.perspective,
                        eps=state.query_tol,
                    )
                )
                if depth < nearest_depth:
                    nearest_depth = depth

            if not np.isfinite(nearest_depth):
                for tri_id in range(state.tri_projected.shape[0]):
                    depth = float(
                        _triangle_depth_value(
                            screen_point=screen_mid,
                            tri_screen=state.tri_projected[tri_id],
                            tri_depths=state.tri_camera_z[tri_id],
                            perspective=state.perspective,
                            eps=state.query_tol,
                        )
                    )
                    if depth < nearest_depth:
                        nearest_depth = depth

            if not np.isfinite(nearest_depth):
                continue

            depth_tol = 1e-8 * max(1.0, abs(edge_depth), abs(nearest_depth))
            if edge_depth > nearest_depth + depth_tol:
                continue

            seg2d = np.vstack(
                (
                    p0 + left * (p1 - p0),
                    p0 + right * (p1 - p0),
                ),
            ).astype(np.float64)

            edge_t0 = _edge_parameter_from_screen_point(
                screen_point=seg2d[0],
                edge_cam0=edge_cam0,
                edge_cam1=edge_cam1,
                perspective=state.perspective,
                eps=state.ray_eps,
            )
            edge_t1 = _edge_parameter_from_screen_point(
                screen_point=seg2d[1],
                edge_cam0=edge_cam0,
                edge_cam1=edge_cam1,
                perspective=state.perspective,
                eps=state.ray_eps,
            )
            if edge_t0 is None or edge_t1 is None:
                continue

            world0 = edge_world0 + edge_t0 * (edge_world1 - edge_world0)
            world1 = edge_world0 + edge_t1 * (edge_world1 - edge_world0)

            visible_segments_2d.append(seg2d)
            visible_segments_3d.append(
                np.vstack((world0, world1)).astype(np.float64),
            )

    return visible_segments_2d, visible_segments_3d


def _init_worker(state: _ProjectionState) -> None:
    """Initialize per-process worker state."""
    global _WORKER_STATE
    _WORKER_STATE = state


def _process_candidate_chunk(candidate_indices: np.ndarray) -> tuple[list[NDArray64], list[NDArray64]]:
    """Process a chunk of candidate edge indices in a worker."""
    if _WORKER_STATE is None:
        raise RuntimeError("Relief worker state has not been initialized.")
    return _extract_from_state_range(_WORKER_STATE, candidate_indices)


def _process_candidate_chunk_with_size(
    work_item: tuple[int, np.ndarray],
) -> tuple[int, int, list[NDArray64], list[NDArray64]]:
    """Process one candidate chunk and return its index, size, and result."""
    chunk_index, candidate_indices = work_item
    seg2d, seg3d = _process_candidate_chunk(candidate_indices)
    return chunk_index, int(candidate_indices.size), seg2d, seg3d


def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """Normalize the requested worker count."""
    if n_jobs is None or n_jobs <= 0:
        return max(os.cpu_count() or 1, 1)
    return max(int(n_jobs), 1)


def _split_candidate_indices(
    n_candidates: int,
    n_jobs: int,
    *,
    progress_mode: str,
) -> list[np.ndarray]:
    """Split candidate edge indices into balanced progress-friendly chunks."""
    if n_candidates == 0:
        return []

    progress_mode = _normalize_progress_mode(progress_mode)
    chunk_updates = max(8 * n_jobs, 32)
    candidate_updates = max(128 * n_jobs, 512)

    if progress_mode == "chunks":
        n_chunks = chunk_updates
    elif progress_mode == "candidates":
        n_chunks = candidate_updates
    else:
        if n_candidates >= 20_000:
            n_chunks = candidate_updates
        elif n_candidates >= 2_000:
            n_chunks = max(32 * n_jobs, 256)
        else:
            n_chunks = chunk_updates

    n_chunks = min(n_candidates, n_chunks)
    chunks = np.array_split(np.arange(n_candidates, dtype=np.int64), n_chunks)
    return [chunk for chunk in chunks if chunk.size > 0]


def _split_serial_candidate_indices(
    n_candidates: int,
    *,
    progress_mode: str,
) -> list[np.ndarray]:
    """Split candidate edges into progress-friendly serial chunks."""
    if n_candidates == 0:
        return []

    progress_mode = _normalize_progress_mode(progress_mode)
    if progress_mode == "chunks":
        n_chunks = 64
    elif progress_mode == "candidates":
        n_chunks = 512
    else:
        n_chunks = 512 if n_candidates >= 20_000 else 128

    n_chunks = min(n_candidates, n_chunks)
    chunks = np.array_split(np.arange(n_candidates, dtype=np.int64), n_chunks)
    return [chunk for chunk in chunks if chunk.size > 0]


def _get_mp_context() -> mp.context.BaseContext:
    """Return a multiprocessing context suitable for relief extraction."""
    methods = mp.get_all_start_methods()
    if "fork" in methods:
        return mp.get_context("fork")
    if "forkserver" in methods:
        return mp.get_context("forkserver")
    return mp.get_context("spawn")


def _extract_serial_chunks(
    state: _ProjectionState,
    chunks: list[np.ndarray],
    *,
    progress: bool,
) -> tuple[list[NDArray64], list[NDArray64]]:
    """Extract visible segments from serial candidate chunks."""
    visible_segments_2d: list[NDArray64] = []
    visible_segments_3d: list[NDArray64] = []
    total = int(sum(int(chunk.size) for chunk in chunks))
    progress_bar = _progress_bar(
        total=total,
        progress=progress,
        desc="visible relief",
    )

    try:
        for chunk in chunks:
            seg2d, seg3d = _extract_from_state_range(
                state=state,
                candidate_indices=chunk,
            )
            visible_segments_2d.extend(seg2d)
            visible_segments_3d.extend(seg3d)
            if progress_bar is not None:
                progress_bar.update(int(chunk.size))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return visible_segments_2d, visible_segments_3d


def extract_visible_relief_from_mesh(
    mesh: PreparedReliefMesh,
    *,
    observer: NDArray64 | tuple[float, float, float],
    target: NDArray64 | tuple[float, float, float],
    up: NDArray64 | tuple[float, float, float] = (0.0, 0.0, 1.0),
    projection: str = "perspective",
    progress: bool = False,
    progress_mode: str = "auto",
    n_jobs: int = 1,
) -> VisibleRelief:
    """Extract visible terrain outlines from a prepared mesh.

    Parameters
    ----------
    mesh
        Prepared terrain mesh returned by ``prepare_relief_mesh``.
    observer
        Camera position in world coordinates.
    target
        Camera target point in world coordinates.
    up
        Approximate world up-direction used to orient the camera.
    projection
        Projection mode, either ``"perspective"`` or ``"orthographic"``.
    progress
        If ``True``, show a ``tqdm`` progress bar while evaluating candidate
        silhouette edges.
    progress_mode
        Progress update granularity. ``"chunks"`` emits a smaller number of
        coarse updates, ``"candidates"`` uses much smaller work chunks for
        smoother progress bars, and ``"auto"`` chooses based on problem size.
    n_jobs
        Number of worker processes used for candidate-edge evaluation.
        ``1`` disables multiprocessing.
    """
    camera = CameraSpec(
        observer=observer,
        target=target,
        up=up,
        projection=projection,
    )
    frame = _build_camera_frame(camera)
    state = _build_projection_state(mesh=mesh, frame=frame)

    visible_segments_2d: list[NDArray64] = []
    visible_segments_3d: list[NDArray64] = []
    total_candidates = int(state.candidate_edges.shape[0])
    progress_mode_norm = _normalize_progress_mode(progress_mode)

    n_jobs_resolved = _resolve_n_jobs(n_jobs)
    if n_jobs_resolved == 1 or total_candidates < 64:
        serial_chunks = (
            _split_serial_candidate_indices(
                total_candidates,
                progress_mode=progress_mode_norm,
            )
            if progress
            else [np.arange(total_candidates, dtype=np.int64)]
        )
        visible_segments_2d, visible_segments_3d = _extract_serial_chunks(
            state=state,
            chunks=serial_chunks,
            progress=progress,
        )
    else:
        chunks = _split_candidate_indices(
            n_candidates=total_candidates,
            n_jobs=n_jobs_resolved,
            progress_mode=progress_mode_norm,
        )
        progress_bar = _progress_bar(
            total=total_candidates,
            progress=progress,
            desc="visible relief",
        )
        try:
            ctx = _get_mp_context()
            chunk_results_2d: list[Optional[list[NDArray64]]] = [
                None for _ in chunks
            ]
            chunk_results_3d: list[Optional[list[NDArray64]]] = [
                None for _ in chunks
            ]
            with ctx.Pool(
                processes=n_jobs_resolved,
                initializer=_init_worker,
                initargs=(state,),
            ) as pool:
                for chunk_index, chunk_size, seg2d, seg3d in pool.imap_unordered(
                    _process_candidate_chunk_with_size,
                    [(idx, chunk) for idx, chunk in enumerate(chunks)],
                    chunksize=1,
                ):
                    chunk_results_2d[chunk_index] = seg2d
                    chunk_results_3d[chunk_index] = seg3d
                    if progress_bar is not None:
                        progress_bar.update(chunk_size)

            for seg2d, seg3d in zip(chunk_results_2d, chunk_results_3d):
                if seg2d is None or seg3d is None:
                    raise RuntimeError(
                        "Parallel relief extraction returned an incomplete "
                        "chunk result."
                    )
                visible_segments_2d.extend(seg2d)
                visible_segments_3d.extend(seg3d)
        except (OSError, RuntimeError, PermissionError):
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None
            visible_segments_2d, visible_segments_3d = _extract_serial_chunks(
                state=state,
                chunks=chunks,
                progress=progress,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()

    merge_tol = 1e-7 * max(
        1.0,
        state.screen_bounds[0][1] - state.screen_bounds[0][0],
        state.screen_bounds[1][1] - state.screen_bounds[1][0],
    )
    polylines = _merge_segments(segments=visible_segments_2d, tol=merge_tol)

    return VisibleRelief(
        polylines=polylines,
        world_segments=visible_segments_3d,
        camera=camera,
        screen_bounds=state.screen_bounds,
    )


def extract_visible_relief(
    x: NDArray64,
    y: NDArray64,
    z: NDArray64,
    *,
    observer: NDArray64 | tuple[float, float, float],
    target: NDArray64 | tuple[float, float, float],
    up: NDArray64 | tuple[float, float, float] = (0.0, 0.0, 1.0),
    projection: str = "perspective",
    xlim: LIM = None,
    ylim: LIM = None,
    progress: bool = False,
    progress_mode: str = "auto",
    n_jobs: int = 1,
) -> VisibleRelief:
    """Extract visible terrain outline polylines from a height field.

    Parameters
    ----------
    x
        1D x-axis values of shape ``(Nx,)``.
    y
        1D y-axis values of shape ``(Ny,)``.
    z
        2D height field of shape ``(Ny, Nx)`` with ``z[y_i, x_j]``.
    observer
        Camera position in world coordinates.
    target
        Camera target point in world coordinates.
    up
        Approximate world up-direction used to orient the camera.
    projection
        Projection mode, either ``"perspective"`` or ``"orthographic"``.
    xlim
        Optional x-range used to crop the field before meshing.
    ylim
        Optional y-range used to crop the field before meshing.
    progress
        If ``True``, show a ``tqdm`` progress bar while evaluating candidate
        silhouette edges.
    progress_mode
        Progress update granularity. ``"chunks"`` emits fewer coarse updates,
        ``"candidates"`` uses much smaller work chunks for smoother updates,
        and ``"auto"`` chooses based on problem size.
    n_jobs
        Number of worker processes used for candidate-edge evaluation.
        ``1`` disables multiprocessing.

    Returns
    -------
    VisibleRelief
        Visible projected polylines, visible source segments, and camera
        metadata.
    """
    mesh = prepare_relief_mesh(x=x, y=y, z=z, xlim=xlim, ylim=ylim)
    return extract_visible_relief_from_mesh(
        mesh=mesh,
        observer=observer,
        target=target,
        up=up,
        projection=projection,
        progress=progress,
        progress_mode=progress_mode,
        n_jobs=n_jobs,
    )
