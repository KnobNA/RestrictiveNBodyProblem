"""Restricted n-body basin-of-attraction simulation on the GPU (CuPy).

Asteroids start at rest at sampled initial positions, move in full n-D under
fixed-planet Newtonian gravity, and stop on a finite-radius collision or timeout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import os

import cupy as cp
import numpy as np
from PIL import Image, ImageDraw

MAX_SUBSTEPS = 24
CFL_SAFETY = 0.4
MIN_STEP = 1e-6


@dataclass
class Planet:
    """A fixed mass in the restricted n-body problem."""

    position: tuple[float, ...]
    mass: float
    radius: float
    color: tuple[int, int, int]


@dataclass
class SlicePlane:
    """A 2-flat in n-D: image pixels spawn only on this plane, then move in D-D."""

    origin: tuple[float, ...]
    axis_u: tuple[float, ...]
    axis_v: tuple[float, ...]
    width: float
    height: float

    @property
    def dim(self) -> int:
        return len(self.origin)

    @classmethod
    def from_view_2d(
        cls,
        view_center: tuple[float, float],
        view_height: float,
        aspect: float,
        dim: int = 2,
    ) -> SlicePlane:
        origin = tuple(view_center) + (0.0,) * (dim - 2)
        axis_u = (1.0,) + (0.0,) * (dim - 1)
        axis_v = (0.0, 1.0) + (0.0,) * (dim - 2)
        return cls(
            origin=origin[:dim],
            axis_u=axis_u[:dim],
            axis_v=axis_v[:dim],
            width=view_height * aspect,
            height=view_height,
        )

    @classmethod
    def from_points(
        cls,
        p0: tuple[float, ...],
        p1: tuple[float, ...],
        p2: tuple[float, ...],
        width: float,
        height: float,
        dim: int | None = None,
    ) -> SlicePlane:
        """Plane through three n-D points. ``p0`` is the origin; ``p1-p0`` and
        ``p2-p0`` set image +x / +y. ``width`` / ``height`` are still the view size.
        """
        d = dim if dim is not None else max(len(p0), len(p1), len(p2))
        a = np.asarray(_pad_vec(p0, d), dtype=np.float64)
        b = np.asarray(_pad_vec(p1, d), dtype=np.float64)
        c = np.asarray(_pad_vec(p2, d), dtype=np.float64)
        return cls(
            origin=tuple(float(x) for x in a),
            axis_u=tuple(float(x) for x in (b - a)),
            axis_v=tuple(float(x) for x in (c - a)),
            width=width,
            height=height,
        )


def _pad_vec(v: tuple[float, ...], dim: int) -> tuple[float, ...]:
    vals = tuple(float(x) for x in v)
    if len(vals) > dim:
        raise ValueError(f"vector length {len(vals)} exceeds dim {dim}")
    return vals + (0.0,) * (dim - len(vals))


def pad_planets(planets: list[Planet], dim: int) -> list[Planet]:
    """Pad planet positions with zeros to length ``dim``."""
    return [
        Planet(
            position=_pad_vec(p.position, dim),
            mass=p.mass,
            radius=p.radius,
            color=p.color,
        )
        for p in planets
    ]


def gram_schmidt(axes: np.ndarray) -> np.ndarray:
    """Orthonormalize rows of ``axes`` (k, D). Raises if they are dependent."""
    out = []
    for raw in np.asarray(axes, dtype=np.float64):
        v = raw.copy()
        for u in out:
            v = v - np.dot(v, u) * u
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            raise ValueError("Cube axes must be linearly independent.")
        out.append(v / nrm)
    return np.stack(out, axis=0)


@dataclass
class NDCube:
    """k-dimensional cube/hypercube in n-D: center C, axes u1..uk, half-extent h.

    The first three axes are the visual 3D cube. Extra axes (u4..) are sampling
    directions for r^k lattices. Coordinates on axis i lie in [-h, h].
    """

    center: tuple[float, ...]
    axes: tuple[tuple[float, ...], ...]
    half_extent: float

    @property
    def k(self) -> int:
        return len(self.axes)

    @property
    def dim(self) -> int:
        return max(len(self.center), max((len(a) for a in self.axes), default=0))

    def arrays(self, dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return padded center (D,) and orthonormal axes (k, D)."""
        d = dim if dim is not None else self.dim
        if d < self.dim:
            raise ValueError("dim cannot be smaller than the cube embedding.")
        if self.k < 1:
            raise ValueError("NDCube needs at least one axis.")
        center = np.asarray(_pad_vec(self.center, d), dtype=np.float64)
        raw = np.stack(
            [np.asarray(_pad_vec(a, d), dtype=np.float64) for a in self.axes],
            axis=0,
        )
        return center, gram_schmidt(raw)

    def project(self, points: np.ndarray, dim: int | None = None) -> np.ndarray:
        """Coordinates d_i = (P - C) · u_i. Shape (N, k)."""
        center, axes = self.arrays(dim=dim)
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] < center.size:
            padded = np.zeros((pts.shape[0], center.size), dtype=np.float64)
            padded[:, : pts.shape[1]] = pts
            pts = padded
        v = pts[:, : center.size] - center
        return v @ axes.T

    def visual_xyz(self, points: np.ndarray, dim: int | None = None) -> np.ndarray:
        """First three cube coordinates as vispy (x, y, z). Shape (N, 3)."""
        d = self.project(points, dim=dim)
        xyz = np.zeros((d.shape[0], 3), dtype=np.float32)
        take = min(3, d.shape[1])
        xyz[:, :take] = d[:, :take]
        return xyz

    def extra_indices(self, points: np.ndarray, resolution: int) -> np.ndarray:
        """Lattice indices 0..r-1 along axes u4..uk. Shape (N, k-3)."""
        if self.k <= 3:
            n = np.asarray(points).shape[0]
            return np.zeros((n, 0), dtype=np.int32)
        return self._lattice_indices(points, resolution, axis_start=3, axis_end=self.k)

    def visual_indices(self, points: np.ndarray, resolution: int) -> np.ndarray:
        """Lattice indices 0..r-1 along visual axes u1..u3. Shape (N, min(3,k))."""
        take = min(3, self.k)
        if take < 1:
            n = np.asarray(points).shape[0]
            return np.zeros((n, 0), dtype=np.int32)
        return self._lattice_indices(points, resolution, axis_start=0, axis_end=take)

    def _lattice_indices(
        self,
        points: np.ndarray,
        resolution: int,
        axis_start: int,
        axis_end: int,
    ) -> np.ndarray:
        n_axes = axis_end - axis_start
        n = np.asarray(points).shape[0]
        if n_axes <= 0:
            return np.zeros((n, 0), dtype=np.int32)
        if resolution < 2:
            return np.zeros((n, n_axes), dtype=np.int32)
        d = self.project(points)
        sl = d[:, axis_start:axis_end]
        span = 2.0 * self.half_extent
        idx = np.rint((sl + self.half_extent) / span * (resolution - 1))
        return np.clip(idx, 0, resolution - 1).astype(np.int32)

    def flat_offset(self, extra_coords: tuple[float, ...]) -> np.ndarray:
        """C' = C + sum t_j u_{3+j} for the current extra-axis slice."""
        center, axes = self.arrays()
        offset = center.copy()
        for j, t in enumerate(extra_coords):
            offset = offset + float(t) * axes[3 + j]
        return offset


def lattice_sample_values(half_extent: float, resolution: int) -> np.ndarray:
    return np.linspace(-half_extent, half_extent, resolution, dtype=np.float64)


def _orthonormal_frame_from_normal(normal: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / (np.linalg.norm(normal) + 1e-30)
    if n.size == 3:
        u = np.cross(up, n)
        if np.linalg.norm(u) < 1e-12:
            alt = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(alt, n)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        return u, v
    u = up - n * np.dot(up, n)
    if np.linalg.norm(u) < 1e-12:
        u = np.zeros_like(n)
        u[int(np.argmin(np.abs(n)))] = 1.0
        u = u - n * np.dot(u, n)
    u = u / np.linalg.norm(u)
    k = (int(np.argmin(np.abs(n))) + 1) % n.size
    w = np.zeros_like(n)
    w[k] = 1.0
    v = w - n * np.dot(w, n) - u * np.dot(w, u)
    v = v / np.linalg.norm(v)
    return u, v


def _unit_axes(axis_u: tuple[float, ...], axis_v: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(axis_u, dtype=np.float64)
    v = np.asarray(axis_v, dtype=np.float64)
    nu = np.linalg.norm(u)
    if nu < 1e-15:
        raise ValueError("axis_u must be non-zero")
    u = u / nu
    v = v - u * np.dot(v, u)
    nv = np.linalg.norm(v)
    if nv < 1e-15:
        raise ValueError("axis_u and axis_v must be linearly independent")
    v = v / nv
    return u, v


def equilateral_triangle_planets(
    *,
    side: float = 1.0,
    mass: float = 1.0,
    radius: float = 0.05,
    colors: tuple[tuple[int, int, int], ...] = (
        (255, 92, 87),
        (91, 192, 222),
        (132, 204, 107),
    ),
    dim: int = 2,
) -> list[Planet]:
    """Three equal masses on an equilateral triangle in the first two coordinates."""
    if dim < 2:
        raise ValueError("dim must be at least 2")
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    xy = (
        (0.0, side * inv_sqrt3),
        (-0.5 * side, -0.5 * side * inv_sqrt3),
        (0.5 * side, -0.5 * side * inv_sqrt3),
    )
    planets = []
    for pos2, color in zip(xy, colors):
        position = pos2 + (0.0,) * (dim - 2)
        planets.append(Planet(position=position[:dim], mass=mass, radius=radius, color=color))
    return planets


def _planet_arrays(planets: list[Planet], dtype=cp.float32):
    dims = {len(p.position) for p in planets}
    if len(dims) != 1:
        raise ValueError("All planets must have the same position dimension.")
    dim = dims.pop()
    pos = cp.asarray([p.position for p in planets], dtype=dtype)
    mass = cp.asarray([p.mass for p in planets], dtype=dtype)
    radius = cp.asarray([p.radius for p in planets], dtype=dtype)
    return dim, pos, mass, radius


def slice_grid_positions(
    width: int,
    height: int,
    plane: SlicePlane,
    dim: int | None = None,
    dtype=cp.float32,
) -> cp.ndarray:
    """One world-space position per pixel, on ``plane`` only. Shape (W*H, D)."""
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive.")
    origin = np.asarray(plane.origin, dtype=np.float64)
    u, v = _unit_axes(plane.axis_u, plane.axis_v)
    if origin.shape != u.shape or origin.shape != v.shape:
        raise ValueError("origin, axis_u, and axis_v must have the same length.")
    dim = dim if dim is not None else int(origin.size)
    if dim < origin.size:
        raise ValueError("dim cannot be smaller than the plane embedding.")

    dw = plane.width / width
    dh = plane.height / height
    us = np.linspace(
        -0.5 * plane.width + 0.5 * dw,
        0.5 * plane.width - 0.5 * dw,
        width,
        dtype=np.float64,
    )
    vs = np.linspace(
        0.5 * plane.height - 0.5 * dh,
        -0.5 * plane.height + 0.5 * dh,
        height,
        dtype=np.float64,
    )
    uu, vv = np.meshgrid(us, vs)
    pos_plane = origin + uu[..., None] * u + vv[..., None] * v
    pos = np.zeros((height, width, dim), dtype=np.float64)
    pos[..., : origin.size] = pos_plane
    return cp.asarray(pos.reshape(height * width, dim), dtype=dtype)


def volume_grid_positions(
    resolution: int,
    box_size: float | tuple[float, ...],
    box_center: tuple[float, ...] | None = None,
    dim: int = 3,
    dtype=cp.float32,
) -> cp.ndarray:
    """Inclusive end-to-end lattice. Shape (r**D, D). Viewer / volume path only."""
    if resolution < 1:
        raise ValueError("resolution must be at least 1.")
    if dim < 1:
        raise ValueError("dim must be at least 1.")
    if np.isscalar(box_size):
        sizes = np.full(dim, float(box_size), dtype=np.float64)
    else:
        sizes = np.asarray(box_size, dtype=np.float64)
        if sizes.shape != (dim,):
            raise ValueError("box_size must be a scalar or a length-D tuple.")
    if box_center is None:
        center = np.zeros(dim, dtype=np.float64)
    else:
        center = np.asarray(box_center, dtype=np.float64)
        if center.shape != (dim,):
            raise ValueError("box_center must have length D.")

    axes = []
    for d in range(dim):
        half = 0.5 * sizes[d]
        axes.append(
            cp.linspace(center[d] - half, center[d] + half, resolution, dtype=dtype)
        )
    mesh = cp.meshgrid(*axes, indexing="ij")
    return cp.stack([m.ravel() for m in mesh], axis=1)


def subspace_grid_positions(
    cube: NDCube,
    resolution: int,
    dim: int | None = None,
    dtype=cp.float32,
) -> cp.ndarray:
    """Inclusive lattice on the cube's k-flat. Shape (r**k, D)."""
    if resolution < 1:
        raise ValueError("resolution must be at least 1.")
    center, axes = cube.arrays(dim=dim)
    k = axes.shape[0]
    d = center.size
    n = resolution ** k
    samples = cp.linspace(
        -cube.half_extent, cube.half_extent, resolution, dtype=dtype
    )
    mesh = cp.meshgrid(*([samples] * k), indexing="ij")
    pos = cp.broadcast_to(cp.asarray(center, dtype=dtype), (n, d)).copy()
    for i in range(k):
        pos = pos + mesh[i].ravel()[:, None] * cp.asarray(axes[i], dtype=dtype)
    return pos


def _acceleration(
    pos: cp.ndarray,
    planet_pos: cp.ndarray,
    planet_mass: cp.ndarray,
    g: float,
    force_exponent: float,
    softening: float,
) -> cp.ndarray:
    """Newtonian acceleration at ``pos``. Inverse-square when force_exponent=3."""
    acc = cp.zeros_like(pos)
    eps2 = softening * softening
    for i in range(planet_pos.shape[0]):
        r = planet_pos[i] - pos
        dist2 = cp.sum(r * r, axis=1, keepdims=True) + eps2
        inv = 1.0 / cp.sqrt(dist2)
        acc += (g * planet_mass[i]) * r * (inv ** force_exponent)
    return acc


def _apply_collisions(
    pos: cp.ndarray,
    alive: cp.ndarray,
    hit: cp.ndarray,
    planet_pos: cp.ndarray,
    planet_radius: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Mark asteroids inside a planet radius. First planet in the list wins ties."""
    for i in range(planet_pos.shape[0]):
        r = planet_pos[i] - pos
        dist2 = cp.sum(r * r, axis=1)
        collided = alive & (dist2 < planet_radius[i] * planet_radius[i])
        hit = cp.where(collided, np.int32(i), hit)
        alive = alive & ~collided
    return alive, hit


def _apply_segment_collisions(
    pos0: cp.ndarray,
    pos1: cp.ndarray,
    alive: cp.ndarray,
    hit: cp.ndarray,
    planet_pos: cp.ndarray,
    planet_radius: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Collide the accepted step segment against each planet sphere."""
    delta = pos1 - pos0
    finite = cp.isfinite(delta).all(axis=1)
    dd = cp.sum(delta * delta, axis=1)
    moving = finite & (dd > 0)
    for i in range(planet_pos.shape[0]):
        w = planet_pos[i] - pos0
        t = cp.sum(w * delta, axis=1)
        t = cp.where(moving, t / (dd + 1e-30), cp.zeros_like(t))
        t = cp.clip(t, 0.0, 1.0)
        closest = pos0 + t[:, None] * delta
        offset = planet_pos[i] - closest
        dist2 = cp.sum(offset * offset, axis=1)
        collided = alive & finite & (dist2 < planet_radius[i] * planet_radius[i])
        hit = cp.where(collided, np.int32(i), hit)
        alive = alive & ~collided
    return alive, hit


def _collide_nonfinite(
    pos_ref: cp.ndarray,
    new_pos: cp.ndarray,
    new_vel: cp.ndarray,
    alive: cp.ndarray,
    hit: cp.ndarray,
    planet_pos: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """If RK4 blew up to Inf/NaN, attribute the hit to the nearest planet."""
    finite = cp.isfinite(new_pos).all(axis=1) & cp.isfinite(new_vel).all(axis=1)
    bad = alive & ~finite
    if not bool(bad.any()):
        return alive, hit
    offset = planet_pos[:, None, :] - pos_ref[None, :, :]
    dist2 = cp.sum(offset * offset, axis=2)
    nearest = cp.argmin(dist2, axis=0).astype(cp.int32)
    hit = cp.where(bad, nearest, hit)
    alive = alive & ~bad
    return alive, hit


def _advance_subset(
    pos: cp.ndarray,
    vel: cp.ndarray,
    time: cp.ndarray,
    alive: cp.ndarray,
    hit: cp.ndarray,
    remaining: cp.ndarray,
    planet_pos: cp.ndarray,
    planet_mass: cp.ndarray,
    planet_radius: cp.ndarray,
    g: float,
    force_exponent: float,
    softening: float,
    cfl_safety: float,
    tiny: np.floating,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """One CFL-limited RK4 step plus collisions on a (already active) subset."""
    acc = _acceleration(pos, planet_pos, planet_mass, g, force_exponent, softening)
    h = _cfl_h(
        pos, vel, acc, planet_pos, planet_radius, remaining, safety=cfl_safety
    )
    new_pos, new_vel = _rk4_step(
        pos,
        vel,
        h,
        planet_pos,
        planet_mass,
        g,
        force_exponent,
        softening,
        acc,
    )
    alive, hit = _apply_segment_collisions(
        pos, new_pos, alive, hit, planet_pos, planet_radius
    )
    alive, hit = _apply_collisions(new_pos, alive, hit, planet_pos, planet_radius)
    alive, hit = _collide_nonfinite(
        pos, new_pos, new_vel, alive, hit, planet_pos
    )
    finite = cp.isfinite(new_pos).all(axis=1) & cp.isfinite(new_vel).all(axis=1)
    write = (h > tiny) & finite
    pos = cp.where(write[:, None], new_pos, pos)
    vel = cp.where(write[:, None], new_vel, vel)
    time = cp.where(write, time + h, time)
    return pos, vel, time, alive, hit


def _cfl_h(
    pos: cp.ndarray,
    vel: cp.ndarray,
    acc: cp.ndarray,
    planet_pos: cp.ndarray,
    planet_radius: cp.ndarray,
    remaining: cp.ndarray,
    safety: float = CFL_SAFETY,
) -> cp.ndarray:
    """Largest step that cannot jump more than a fraction of the gap to a surface."""
    dist_surf = cp.full(pos.shape[0], np.float32(np.inf), dtype=pos.dtype)
    for i in range(planet_pos.shape[0]):
        r = cp.sqrt(cp.sum((planet_pos[i] - pos) ** 2, axis=1))
        gap = r - planet_radius[i]
        floor = 0.25 * planet_radius[i]
        dist_surf = cp.minimum(dist_surf, cp.maximum(gap, floor))
    speed = cp.sqrt(cp.sum(vel * vel, axis=1))
    acc_mag = cp.sqrt(cp.sum(acc * acc, axis=1))
    denom = speed + cp.sqrt(2.0 * acc_mag * dist_surf + 1e-30) + 1e-12
    h_cfl = safety * dist_surf / denom
    h = cp.minimum(remaining, h_cfl)
    return cp.maximum(h, np.float32(0))


def _rk4_step(
    pos: cp.ndarray,
    vel: cp.ndarray,
    h: cp.ndarray,
    planet_pos: cp.ndarray,
    planet_mass: cp.ndarray,
    g: float,
    force_exponent: float,
    softening: float,
    k1_v: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """RK4 with a per-particle step ``h`` of shape (N,). ``k1_v`` is acc(pos)."""
    ht = h[:, None]
    half = 0.5 * ht
    k1_x = vel
    k2_v = _acceleration(
        pos + half * k1_x, planet_pos, planet_mass, g, force_exponent, softening
    )
    k2_x = vel + half * k1_v
    k3_v = _acceleration(
        pos + half * k2_x, planet_pos, planet_mass, g, force_exponent, softening
    )
    k3_x = vel + half * k2_v
    k4_v = _acceleration(
        pos + ht * k3_x, planet_pos, planet_mass, g, force_exponent, softening
    )
    k4_x = vel + ht * k3_v
    sixth = ht / 6.0
    new_pos = pos + sixth * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    new_vel = vel + sixth * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return new_pos, new_vel


def integrate_asteroids(
    pos: cp.ndarray,
    planets: list[Planet],
    *,
    g: float = 1.0,
    dt: float = 0.02,
    t_max: float = 40.0,
    force_exponent: float = 3.0,
    softening: float = 1e-6,
    progress_every: int = 50,
    cfl_safety: float = CFL_SAFETY,
    max_substeps: int = MAX_SUBSTEPS,
) -> cp.ndarray:
    """RK4-integrate asteroids from rest until collision or ``t_max``.

    Does not modify the input ``pos`` array; the viewer needs those initial
    lattice coordinates to draw basins.

    Each asteroid has its own clock. Far from planets it advances by ``dt``;
    near a surface the step is CFL-limited so the path cannot skip the sphere.
    Hits are counted only when the body overlaps a planet or the accepted
    segment intersects it — never by predicted periapsis. Collided and
    timed-out asteroids are dropped from the GPU work set each iteration.
    """
    if not planets:
        raise ValueError("Need at least one planet.")

    dim, planet_pos, planet_mass, planet_radius = _planet_arrays(planets, dtype=pos.dtype)
    if pos.ndim != 2 or pos.shape[1] != dim:
        raise ValueError(f"pos must have shape (N, {dim}) to match the planets.")

    # Work on a copy so the caller's initial lattice is not overwritten.
    pos = pos.copy()

    n = pos.shape[0]
    vel = cp.zeros_like(pos)
    alive = cp.ones(n, dtype=cp.bool_)
    hit = cp.full(n, -1, dtype=cp.int32)
    time = cp.zeros(n, dtype=pos.dtype)
    tiny = np.float32(MIN_STEP)

    alive, hit = _apply_collisions(pos, alive, hit, planet_pos, planet_radius)

    max_steps = max(1, math.ceil(t_max / dt))
    max_iters = max_steps * max_substeps
    print(
        f"RK4+CFL: {n:,} asteroids, t_max={t_max}, dt<={dt}, "
        f"max_iters={max_iters}"
    )

    for iteration in range(1, max_iters + 1):
        remaining = cp.where(alive, np.float32(t_max) - time, np.float32(0))
        remaining = cp.minimum(remaining, np.float32(dt))
        remaining = cp.maximum(remaining, np.float32(0))
        active = alive & (remaining > tiny)
        idx = cp.nonzero(active)[0]
        n_active = int(idx.size)

        if progress_every and (iteration == 1 or iteration % progress_every == 0):
            n_alive = int(alive.sum())
            if n_alive == 0:
                print(f"iter {iteration - 1}/{max_iters}  all asteroids resolved")
                break
            t_mean = float(time[alive].mean()) if n_alive else t_max
            print(
                f"iter {iteration}/{max_iters}  "
                f"alive={n_alive:,}/{n:,} ({100.0 * n_alive / n:.1f}%)  "
                f"active={n_active:,}  "
                f"t_mean={t_mean:.3f}/{t_max}"
            )
        if n_active == 0:
            break

        if n_active == n:
            pos, vel, time, alive, hit = _advance_subset(
                pos,
                vel,
                time,
                alive,
                hit,
                remaining,
                planet_pos,
                planet_mass,
                planet_radius,
                g,
                force_exponent,
                softening,
                cfl_safety,
                tiny,
            )
        else:
            p, v, t, a, h = _advance_subset(
                pos[idx],
                vel[idx],
                time[idx],
                alive[idx],
                hit[idx],
                remaining[idx],
                planet_pos,
                planet_mass,
                planet_radius,
                g,
                force_exponent,
                softening,
                cfl_safety,
                tiny,
            )
            pos[idx] = p
            vel[idx] = v
            time[idx] = t
            alive[idx] = a
            hit[idx] = h
    else:
        n_alive = int(alive.sum())
        print(
            f"iter {max_iters}/{max_iters}  "
            f"timed out={n_alive:,}/{n:,} ({100.0 * n_alive / n:.1f}% unresolved)"
        )

    n_alive = int(alive.sum())
    if n_alive:
        print(
            f"done  timed out={n_alive:,}/{n:,} ({100.0 * n_alive / n:.1f}% unresolved)"
        )
    return hit


def _plane_to_pixel(
    point: tuple[float, ...],
    plane: SlicePlane,
    width: int,
    height: int,
) -> tuple[float, float, float]:
    """Return (pixel_x, pixel_y, distance from the point to the 2-flat in full D)."""
    dim = max(len(plane.origin), len(plane.axis_u), len(plane.axis_v), len(point))
    origin = np.asarray(_pad_vec(tuple(float(x) for x in plane.origin), dim), dtype=np.float64)
    u, v = _unit_axes(
        _pad_vec(tuple(float(x) for x in plane.axis_u), dim),
        _pad_vec(tuple(float(x) for x in plane.axis_v), dim),
    )
    p = np.asarray(_pad_vec(tuple(float(x) for x in point), dim), dtype=np.float64)
    rel = p - origin
    cu = float(np.dot(rel, u))
    cv = float(np.dot(rel, v))
    dist = float(np.linalg.norm(rel - cu * u - cv * v))
    px = (cu + 0.5 * plane.width) / plane.width * width
    py = (0.5 * plane.height - cv) / plane.height * height
    return px, py, dist


def colorize_hits(
    hit: cp.ndarray,
    planets: list[Planet],
    width: int,
    height: int,
) -> np.ndarray:
    """Map collision indices to an RGB uint8 image. Unhit pixels stay black."""
    palette = cp.zeros((len(planets) + 1, 3), dtype=cp.uint8)
    for i, planet in enumerate(planets):
        palette[i + 1] = cp.asarray(planet.color, dtype=cp.uint8)
    rgb = palette[hit + 1].reshape(height, width, 3)
    return cp.asnumpy(rgb)


def draw_planet_rings(
    rgb: np.ndarray,
    planets: list[Planet],
    plane: SlicePlane,
    ring_color: tuple[int, int, int] = (255, 255, 255),
    width_px: int | None = None,
) -> np.ndarray:
    """Outline each n-sphere's circular cross-section with the 2-flat.

    Distance is computed in full ambient dimension. Drawn only when
    ``dist < radius``; ring radius is ``sqrt(R^2 - dist^2)``.
    """
    height, width = rgb.shape[:2]
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    if width_px is None:
        width_px = max(1, round(height / 540))
    for planet in planets:
        px, py, dist = _plane_to_pixel(planet.position, plane, width, height)
        if dist >= planet.radius:
            continue
        r_world = math.sqrt(planet.radius * planet.radius - dist * dist)
        r_px = r_world / plane.height * height
        if r_px < 0.5:
            draw.point((px, py), fill=ring_color)
            continue
        bbox = [px - r_px, py - r_px, px + r_px, py + r_px]
        draw.ellipse(bbox, outline=ring_color, width=max(1, width_px))
    return np.asarray(image)


def render_basins(
    planets: list[Planet],
    width: int,
    height: int,
    *,
    plane: SlicePlane | None = None,
    g: float = 1.0,
    dt: float = 0.02,
    t_max: float = 40.0,
    view_center: tuple[float, float] = (0.0, 0.0),
    view_height: float = 2.5,
    force_exponent: float = 3.0,
    softening: float = 1e-6,
    progress_every: int = 50,
    draw_rings: bool = True,
) -> np.ndarray:
    """Simulate one asteroid per pixel on ``plane`` (only). Returns RGB (H, W, 3)."""
    if not planets:
        raise ValueError("Need at least one planet.")
    dim = max(len(planets[0].position), plane.dim if plane is not None else 0)
    for planet in planets:
        dim = max(dim, len(planet.position))
    planets = pad_planets(planets, dim)
    if plane is None:
        plane = SlicePlane.from_view_2d(
            view_center, view_height, width / height, dim=dim
        )
    else:
        plane = SlicePlane(
            origin=_pad_vec(plane.origin, dim),
            axis_u=_pad_vec(plane.axis_u, dim),
            axis_v=_pad_vec(plane.axis_v, dim),
            width=plane.width,
            height=plane.height,
        )

    n = width * height
    print(f"Rendering {width}x{height} slice ({n:,} asteroids on the plane, D={dim})")
    pos = slice_grid_positions(width, height, plane, dim=dim)
    hit = integrate_asteroids(
        pos,
        planets,
        g=g,
        dt=dt,
        t_max=t_max,
        force_exponent=force_exponent,
        softening=softening,
        progress_every=progress_every,
    )
    rgb = colorize_hits(hit, planets, width, height)
    if draw_rings:
        rgb = draw_planet_rings(rgb, planets, plane)
    return rgb


def simulate_volume(
    planets: list[Planet],
    resolution: int,
    cube: NDCube,
    *,
    g: float = 1.0,
    dt: float = 0.02,
    t_max: float = 40.0,
    force_exponent: float = 3.0,
    softening: float = 1e-6,
    progress_every: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """k-flat lattice for the viewer. Returns (initial_pos, hit) on the CPU."""
    if not planets:
        raise ValueError("Need at least one planet.")
    dim = max(len(p.position) for p in planets)
    dim = max(dim, cube.dim)
    planets = pad_planets(planets, dim)
    n = resolution ** cube.k
    if n >= 5_000_000:
        print(
            f"Warning: r={resolution}, k={cube.k} -> {n:,} asteroids. "
            "This may take a long time and a lot of GPU memory."
        )
    pos = subspace_grid_positions(cube, resolution, dim=dim)
    print(
        f"Subspace grid r={resolution} k={cube.k} D={dim} -> {pos.shape[0]:,} "
        f"asteroids  h={cube.half_extent}"
    )
    hit = integrate_asteroids(
        pos,
        planets,
        g=g,
        dt=dt,
        t_max=t_max,
        force_exponent=force_exponent,
        softening=softening,
        progress_every=progress_every,
    )
    return cp.asnumpy(pos), cp.asnumpy(hit)


def planet_3flat_ball(
    planet: Planet,
    cube: NDCube,
    extra_coords: tuple[float, ...],
) -> tuple[tuple[float, float, float], float] | None:
    """Intersection of an n-sphere with the current visual 3-flat.

    Returns ((x,y,z) in cube coordinates, apparent_radius) or None if the
    hypersphere misses the 3-flat.
    """
    center, axes = cube.arrays()
    offset = cube.flat_offset(extra_coords)
    p = np.asarray(_pad_vec(planet.position, center.size), dtype=np.float64)
    v = p - offset
    d = axes[: min(3, axes.shape[0])] @ v
    xyz = np.zeros(3, dtype=np.float64)
    xyz[: d.size] = d
    orth2 = float(np.dot(v, v) - np.dot(d, d))
    orth2 = max(orth2, 0.0)
    if orth2 >= planet.radius * planet.radius:
        return None
    apparent = math.sqrt(planet.radius * planet.radius - orth2)
    return (float(xyz[0]), float(xyz[1]), float(xyz[2])), float(apparent)


def save_basins_png(rgb: np.ndarray, path: str | Path) -> Path:
    """Write an RGB array to PNG, creating parent directories as needed."""
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    Image.fromarray(rgb, mode="RGB").save(tmp)
    try:
        tmp.replace(path)
    except OSError:
        path = path.with_stem(f"{path.stem}_{os.getpid()}")
        tmp.replace(path)
    print(f"Wrote {path}")
    return path
