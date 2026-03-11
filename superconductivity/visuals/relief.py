"""Visible relief extraction for regular height fields.

This module turns a regular scalar field ``z[y_i, x_j]`` into a set of
view-dependent visible outline polylines. The terrain is treated as a
triangulated top surface, projected from an explicit camera, and reduced to
the subset of silhouette segments that remain visible after hidden-line
removal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from superconductivity.utilities.types import LIM, NDArray64

from .helper import check_xyz, get_xylim_indices


def _as_point3(value: NDArray64 | tuple[float, float, float]) -> NDArray64:
    """Convert a 3-vector-like input to a float64 NumPy array.

    Parameters
    ----------
    value
        Input point or vector with exactly three components.

    Returns
    -------
    ndarray
        Array of shape ``(3,)`` with dtype ``float64``.

    Raises
    ------
    ValueError
        If the input does not describe a 3-vector.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("Expected a 3-vector with shape (3,).")
    return arr


def _normalize(vec: NDArray64, *, name: str) -> NDArray64:
    """Return a unit-length copy of ``vec``.

    Parameters
    ----------
    vec
        Input vector.
    name
        Human-readable vector name used in error messages.

    Returns
    -------
    ndarray
        Normalized vector.

    Raises
    ------
    ValueError
        If the vector norm is numerically zero.
    """
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return vec / norm


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
class _CameraFrame:
    """Derived camera basis and projection data."""

    spec: CameraSpec
    right: NDArray64
    up: NDArray64
    forward: NDArray64


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
    """Project world-space vertices into camera screen coordinates."""
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

    return projected.astype(np.float64), depths.astype(np.float64)


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


def _face_visibility(
    tri_vertices: NDArray64,
    camera: _CameraFrame,
) -> NDArray64:
    """Compute signed face visibility values.

    Positive values indicate front-facing triangles.
    """
    v0 = tri_vertices[:, 0, :]
    v1 = tri_vertices[:, 1, :]
    v2 = tri_vertices[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0

    if camera.spec.projection == "perspective":
        view = camera.spec.observer[None, :] - centroids
    else:
        view = np.broadcast_to(-camera.forward, normals.shape)

    return np.einsum("ij,ij->i", normals, view)


def _classify_sign(values: NDArray64, eps: float) -> np.ndarray:
    """Classify signed values into ``{-1, 0, +1}`` with tolerance."""
    out = np.zeros(values.shape, dtype=np.int8)
    out[values > eps] = 1
    out[values < -eps] = -1
    return out


def _collect_candidate_edges(
    triangles: np.ndarray,
    face_values: NDArray64,
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Collect silhouette candidate edges from mesh adjacency."""
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_idx, tri in enumerate(triangles):
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(key, []).append(face_idx)

    face_eps = 1e-12 * max(1.0, float(np.max(np.abs(face_values))))
    face_sign = _classify_sign(face_values, eps=face_eps)

    candidates: list[tuple[int, int, tuple[int, ...]]] = []
    for (u, v), faces in edge_to_faces.items():
        if len(faces) == 1:
            if face_sign[faces[0]] >= 0:
                candidates.append((u, v, tuple(faces)))
            continue

        if len(faces) != 2:
            continue

        s0 = int(face_sign[faces[0]])
        s1 = int(face_sign[faces[1]])
        if s0 == s1 and s0 != 0:
            continue
        if s0 == 0 and s1 == 0:
            continue
        candidates.append((u, v, tuple(faces)))

    return candidates


def _segment_intersection_parameter(
    p0: NDArray64,
    p1: NDArray64,
    q0: NDArray64,
    q1: NDArray64,
    tol: float,
) -> Optional[float]:
    """Return the parameter of a proper 2D segment intersection on ``p``."""
    r = p1 - p0
    s = q1 - q0
    denom = r[0] * s[1] - r[1] * s[0]
    qp = q0 - p0

    if abs(float(denom)) <= tol:
        return None

    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom

    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        return float(np.clip(t, 0.0, 1.0))
    return None


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


def _ray_triangle_intersections(
    origin: NDArray64,
    direction: NDArray64,
    tri_v0: NDArray64,
    tri_e1: NDArray64,
    tri_e2: NDArray64,
    eps: float,
) -> NDArray64:
    """Intersect a ray with many triangles via Moller-Trumbore."""
    pvec = np.cross(direction[None, :], tri_e2)
    det = np.einsum("ij,ij->i", tri_e1, pvec)

    mask = np.abs(det) > eps
    if not np.any(mask):
        return np.empty(0, dtype=np.float64)

    inv_det = np.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]

    tvec = origin[None, :] - tri_v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, tri_e1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    t = np.einsum("ij,ij->i", tri_e2, qvec) * inv_det

    mask &= u >= -eps
    mask &= v >= -eps
    mask &= (u + v) <= 1.0 + eps
    mask &= t > eps

    return t[mask].astype(np.float64)


def _ray_face_intersection(
    origin: NDArray64,
    direction: NDArray64,
    tri_vertices: NDArray64,
    eps: float,
) -> tuple[Optional[float], Optional[NDArray64]]:
    """Intersect a ray with a single triangle and return depth and point."""
    v0 = tri_vertices[0]
    e1 = tri_vertices[1] - v0
    e2 = tri_vertices[2] - v0

    hits = _ray_triangle_intersections(
        origin=origin,
        direction=direction,
        tri_v0=v0[None, :],
        tri_e1=e1[None, :],
        tri_e2=e2[None, :],
        eps=eps,
    )
    if hits.size == 0:
        return None, None

    depth = float(hits[0])
    point = origin + depth * direction
    return depth, point.astype(np.float64)


def _screen_ray(
    point: NDArray64,
    camera: _CameraFrame,
) -> tuple[NDArray64, NDArray64]:
    """Build the world-space ray corresponding to a screen-space point."""
    if camera.spec.projection == "perspective":
        origin = camera.spec.observer
        direction = (
            camera.forward
            + point[0] * camera.right
            + point[1] * camera.up
        )
    else:
        origin = (
            camera.spec.observer
            + point[0] * camera.right
            + point[1] * camera.up
        )
        direction = camera.forward

    return origin.astype(np.float64), direction.astype(np.float64)


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

    Returns
    -------
    VisibleRelief
        Visible projected polylines, visible source segments, and camera
        metadata.
    """
    x_arr, y_arr, z_arr = check_xyz(x=x, y=y, z=z)
    ix, iy = get_xylim_indices(x=x_arr, y=y_arr, xlim=xlim, ylim=ylim)
    x_sel = np.asarray(x_arr[ix], dtype=np.float64)
    y_sel = np.asarray(y_arr[iy], dtype=np.float64)
    z_sel = np.asarray(z_arr[np.ix_(iy, ix)], dtype=np.float64)

    camera = CameraSpec(
        observer=observer,
        target=target,
        up=up,
        projection=projection,
    )
    frame = _build_camera_frame(camera)

    vertices, triangles = _build_mesh(x=x_sel, y=y_sel, z=z_sel)
    projected_vertices, _ = _project_vertices(vertices=vertices, camera=frame)

    tri_vertices = vertices[triangles]
    tri_projected = projected_vertices[triangles]
    face_values = _face_visibility(tri_vertices=tri_vertices, camera=frame)
    candidate_edges = _collect_candidate_edges(
        triangles=triangles,
        face_values=face_values,
    )

    tri_v0 = tri_vertices[:, 0, :]
    tri_e1 = tri_vertices[:, 1, :] - tri_v0
    tri_e2 = tri_vertices[:, 2, :] - tri_v0

    boundary_segments = tri_projected[:, ((0, 1), (1, 2), (2, 0)), :]
    boundary_segments = boundary_segments.reshape(-1, 2, 2)
    boundary_min = np.min(boundary_segments, axis=1)
    boundary_max = np.max(boundary_segments, axis=1)

    screen_min = np.min(projected_vertices, axis=0)
    screen_max = np.max(projected_vertices, axis=0)
    screen_span = np.maximum(screen_max - screen_min, 1.0)
    split_tol = 1e-9 * float(np.max(screen_span))
    ray_eps = 1e-10
    visible_segments_2d: list[NDArray64] = []
    visible_segments_3d: list[NDArray64] = []

    for vertex_a, vertex_b, faces in candidate_edges:
        p0 = projected_vertices[vertex_a]
        p1 = projected_vertices[vertex_b]

        seg_min = np.minimum(p0, p1) - split_tol
        seg_max = np.maximum(p0, p1) + split_tol
        overlap = np.all(boundary_max >= seg_min, axis=1)
        overlap &= np.all(boundary_min <= seg_max, axis=1)

        params = [0.0, 1.0]
        for q0, q1 in boundary_segments[overlap]:
            t_hit = _segment_intersection_parameter(
                p0=p0,
                p1=p1,
                q0=q0,
                q1=q1,
                tol=split_tol,
            )
            if t_hit is not None:
                params.append(t_hit)

        support_face = max(faces, key=lambda idx: float(face_values[idx]))
        params_unique = _unique_parameters(params=params, tol=split_tol)

        for left, right in zip(params_unique[:-1], params_unique[1:]):
            if right - left <= split_tol:
                continue

            mid_param = 0.5 * (left + right)
            screen_mid = p0 + mid_param * (p1 - p0)
            ray_origin, ray_direction = _screen_ray(point=screen_mid, camera=frame)

            edge_depth, _ = _ray_face_intersection(
                origin=ray_origin,
                direction=ray_direction,
                tri_vertices=tri_vertices[support_face],
                eps=ray_eps,
            )
            if edge_depth is None:
                continue

            all_depths = _ray_triangle_intersections(
                origin=ray_origin,
                direction=ray_direction,
                tri_v0=tri_v0,
                tri_e1=tri_e1,
                tri_e2=tri_e2,
                eps=ray_eps,
            )
            if all_depths.size == 0:
                continue

            nearest_depth = float(np.min(all_depths))
            depth_tol = 1e-8 * max(1.0, abs(edge_depth), abs(nearest_depth))
            if edge_depth > nearest_depth + depth_tol:
                continue

            seg2d = np.vstack(
                (
                    p0 + left * (p1 - p0),
                    p0 + right * (p1 - p0),
                ),
            ).astype(np.float64)

            world_points: list[NDArray64] = []
            for endpoint in seg2d:
                end_origin, end_direction = _screen_ray(
                    point=endpoint,
                    camera=frame,
                )
                _, world_point = _ray_face_intersection(
                    origin=end_origin,
                    direction=end_direction,
                    tri_vertices=tri_vertices[support_face],
                    eps=ray_eps,
                )
                if world_point is None:
                    world_points = []
                    break
                world_points.append(world_point)

            if len(world_points) != 2:
                continue

            visible_segments_2d.append(seg2d)
            visible_segments_3d.append(
                np.vstack(world_points).astype(np.float64),
            )

    merge_tol = 1e-7 * float(np.max(screen_span))
    polylines = _merge_segments(segments=visible_segments_2d, tol=merge_tol)

    screen_bounds = (
        (float(screen_min[0]), float(screen_max[0])),
        (float(screen_min[1]), float(screen_max[1])),
    )

    return VisibleRelief(
        polylines=polylines,
        world_segments=visible_segments_3d,
        camera=camera,
        screen_bounds=screen_bounds,
    )
