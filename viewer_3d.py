"""Interactive 3D view of an n-D hypercube lattice of collision basins.

The white box is the visual 3-cube (first three axes of NDCube). Extra axes
are sampled as r^k points; sliders pick which 3-flat to show. Coordinate-plane
toggles pick the union of lattice slices parallel to yz, xz, and xy.

    python viewer_3d.py
    python viewer_3d.py output/my_basins.npz
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from vispy import scene, use
from vispy.visuals.transforms import STTransform

use("glfw")

from simulation import (
    PLANETS,
    NDCube,
    Planet,
    lattice_sample_values,
    planet_3flat_ball,
    simulate_volume,
    time_to_hit_rgb,
)

# --- knobs (new simulation only; ignored when opening a saved model) ---
RESOLUTION = 50
HALF_EXTENT = 1.25
CUBE_CENTER = (0.0, 0.0, 0.0)
# Exactly k vectors. First three are the visible cube; further vectors add
# extra sampling dimensions (and sliders). Example 4D: add (0, 0, 0, 1).
CUBE_AXES = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 1.0),
)
G = 1.0
DT = 0.02
T_MAX = 40.0
FORCE_EXPONENT = 3.0
SOFTENING = 1e-6
RELATIVISTIC = False
C_LIGHT = 10.0
POINT_SIZE = 4.0
HOVER_DIM_ALPHA = 0.12

MODEL_VERSION = 2
FILE_TYPES = [("N-body basin model", "*.npz"), ("All files", "*.*")]

_PLANE_FACE_LABELS = (
    ("X (yz planes)", 0),
    ("Y (xz planes)", 1),
    ("Z (xy planes)", 2),
)


def _rgb01(color: tuple[int, int, int]) -> tuple[float, float, float, float]:
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0)


def _planets_to_meta(planets: list[Planet]) -> list[dict]:
    return [
        {
            "position": [float(x) for x in p.position],
            "mass": float(p.mass),
            "radius": float(p.radius),
            "color": [int(c) for c in p.color],
        }
        for p in planets
    ]


def _planets_from_meta(raw: list[dict]) -> list[Planet]:
    return [
        Planet(
            position=tuple(float(x) for x in item["position"]),
            mass=float(item["mass"]),
            radius=float(item["radius"]),
            color=tuple(int(c) for c in item["color"]),
        )
        for item in raw
    ]


def _cube_from_legacy(meta: dict) -> NDCube:
    box_size = meta["box_size"]
    if isinstance(box_size, list):
        size = float(box_size[0])
    else:
        size = float(box_size)
    center = tuple(float(x) for x in meta["box_center"])
    if len(center) < 3:
        center = center + (0.0,) * (3 - len(center))
    return NDCube(
        center=center[:3],
        axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        half_extent=0.5 * size,
    )


def save_model(
    path: str | Path,
    *,
    positions: np.ndarray,
    hit: np.ndarray,
    planets: list[Planet],
    cube: NDCube,
    resolution: int,
    point_size: float = POINT_SIZE,
    time: np.ndarray | None = None,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": MODEL_VERSION,
        "resolution": int(resolution),
        "half_extent": float(cube.half_extent),
        "cube_center": [float(x) for x in cube.center],
        "cube_axes": [[float(x) for x in ax] for ax in cube.axes],
        "point_size": float(point_size),
        "planets": _planets_to_meta(planets),
    }
    if extra:
        meta.update(extra)
    payload: dict = {
        "positions": np.asarray(positions, dtype=np.float32),
        "hit": np.asarray(hit, dtype=np.int32),
        "meta": np.asarray(json.dumps(meta)),
    }
    if time is not None:
        payload["time"] = np.asarray(time, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path


def load_model(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "positions" not in data or "hit" not in data or "meta" not in data:
            raise ValueError(f"Not a basin model file: {path}")
        meta = json.loads(str(data["meta"]))
        positions = np.asarray(data["positions"], dtype=np.float32)
        hit = np.asarray(data["hit"], dtype=np.int32)
        time = None
        if "time" in data:
            time = np.asarray(data["time"], dtype=np.float32)
    version = int(meta.get("version", 1))
    meta.setdefault("relativistic", False)
    meta.setdefault("c_light", 10.0)
    if version >= 2:
        cube = NDCube(
            center=tuple(float(x) for x in meta["cube_center"]),
            axes=tuple(tuple(float(x) for x in ax) for ax in meta["cube_axes"]),
            half_extent=float(meta["half_extent"]),
        )
    else:
        cube = _cube_from_legacy(meta)
    return {
        "positions": positions,
        "hit": hit,
        "time": time,
        "t_max": float(meta.get("t_max", T_MAX)),
        "planets": _planets_from_meta(meta["planets"]),
        "cube": cube,
        "resolution": int(meta.get("resolution", 0)),
        "point_size": float(meta.get("point_size", POINT_SIZE)),
        "meta": meta,
        "path": path,
    }


def _clamp_slider(value, resolution: int) -> int:
    """Map a Tk scale value to a 1-based lattice index in 1..r."""
    r = max(int(resolution), 1)
    try:
        raw = int(round(float(value)))
    except (TypeError, ValueError):
        raw = 1
    return int(np.clip(raw, 1, r))


def _mid_extra_slider(resolution: int, half_extent: float) -> int:
    """1-based slider whose lattice coordinate is nearest 0."""
    r = max(int(resolution), 1)
    samples = lattice_sample_values(half_extent, r)
    return int(np.argmin(np.abs(samples))) + 1


def _extra_coords(cube: NDCube, resolution: int, slider_vals: list[int]) -> tuple[float, ...]:
    samples = lattice_sample_values(cube.half_extent, resolution)
    coords = []
    for s in slider_vals:
        idx = _clamp_slider(s, resolution) - 1
        coords.append(float(samples[idx]) if resolution >= 1 else 0.0)
    return tuple(coords)


def _build_scene(
    positions: np.ndarray,
    hit: np.ndarray,
    planets: list[Planet],
    cube: NDCube,
    resolution: int,
    point_size: float = POINT_SIZE,
    title: str = "Restricted n-body basins (3D)",
):
    h = cube.half_extent
    side = 2.0 * h
    canvas = scene.SceneCanvas(
        keys="interactive",
        bgcolor="black",
        size=(1100, 800),
        title=title,
        show=True,
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.ArcballCamera(
        fov=45,
        distance=side * 2.2,
        center=(0.0, 0.0, 0.0),
    )
    view.camera.scale_factor = side

    box = scene.visuals.Box(
        width=side,
        height=side,
        depth=side,
        color=(0, 0, 0, 0),
        edge_color="white",
        parent=view.scene,
    )
    box.mesh.visible = False

    xyz = cube.visual_xyz(positions)
    extra_idx = cube.extra_indices(positions, resolution)
    visual_idx = cube.visual_indices(positions, resolution)
    hit = np.asarray(hit)

    planet_meshes = []
    for planet in planets:
        r, g, b, _a = _rgb01(planet.color)
        fill = scene.visuals.Sphere(
            radius=1.0,
            rows=16,
            cols=24,
            method="latitude",
            color=(r, g, b, 1.0),
            edge_color=None,
            parent=view.scene,
        )
        fill.mesh.set_gl_state(
            polygon_offset_fill=True,
            polygon_offset=(1, 8),
            depth_test=True,
        )
        fill.transform = STTransform(translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
        wire = scene.visuals.Sphere(
            radius=1.0,
            rows=16,
            cols=24,
            method="latitude",
            color=(0.0, 0.0, 0.0, 0.0),
            edge_color=(1.0, 1.0, 1.0, 1.0),
            parent=view.scene,
        )
        wire.mesh.visible = False
        wire.border.visible = True
        wire.transform = STTransform(translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
        planet_meshes.append((fill, wire))

    hit_markers = []
    for i, planet in enumerate(planets):
        markers = scene.visuals.Markers(parent=view.scene)
        markers.set_data(
            np.zeros((0, 3), dtype=np.float32),
            face_color=_rgb01(planet.color),
            edge_width=0,
            size=point_size,
            symbol="o",
        )
        markers.antialias = 0
        markers.set_gl_state(depth_test=True, blend=True)
        hit_markers.append(markers)

    return canvas, planet_meshes, hit_markers, xyz, extra_idx, visual_idx, hit


def _layer_mask(
    extra_idx: np.ndarray,
    slider_vals: list[int],
    resolution: int,
) -> np.ndarray:
    if extra_idx.shape[1] == 0:
        return np.ones(extra_idx.shape[0], dtype=bool)
    mask = np.ones(extra_idx.shape[0], dtype=bool)
    for j, s in enumerate(slider_vals):
        mask &= extra_idx[:, j] == (_clamp_slider(s, resolution) - 1)
    return mask


def _union_plane_mask(visual_idx: np.ndarray, enabled: list[np.ndarray]) -> np.ndarray:
    n = visual_idx.shape[0]
    n_axes = visual_idx.shape[1]
    if n_axes == 0:
        return np.ones(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    for axis, on in enumerate(enabled):
        if axis >= n_axes:
            break
        mask |= np.asarray(on, dtype=bool)[visual_idx[:, axis]]
    return mask


class PlaneStripPicker(ttk.Frame):
    """Three face diagrams: r clickable vertical lines per visual axis."""

    def __init__(
        self,
        parent,
        *,
        resolution: int,
        n_visual: int,
        on_change,
        size: int = 108,
    ):
        super().__init__(parent)
        self.resolution = max(int(resolution), 1)
        self.n_visual = max(int(n_visual), 0)
        self.on_change = on_change
        self.size = size
        self.enabled = [
            np.ones(self.resolution, dtype=bool) for _ in range(self.n_visual)
        ]
        self.hover: tuple[int, int] | None = None
        self._canvases: list[tk.Canvas] = []

        faces = ttk.Frame(self)
        faces.pack(fill="x")
        for label, axis in _PLANE_FACE_LABELS:
            col = ttk.Frame(faces)
            col.pack(side="left", expand=True, fill="both", padx=2)
            ttk.Label(col, text=label, anchor="center").pack(fill="x")
            canvas = tk.Canvas(
                col,
                width=size,
                height=size,
                bg="#1a1a1a",
                highlightthickness=1,
                highlightbackground="#555555",
            )
            canvas.pack(pady=(2, 0))
            self._canvases.append(canvas)
            if axis < self.n_visual:
                canvas.bind("<Button-1>", lambda e, a=axis: self._on_click(a, e))
                canvas.bind("<Motion>", lambda e, a=axis: self._on_motion(a, e))
                canvas.bind("<Leave>", lambda _e, a=axis: self._on_leave(a))
            else:
                canvas.create_text(
                    size / 2,
                    size / 2,
                    text="n/a",
                    fill="#666666",
                )

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="All on", command=self.all_on).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ttk.Button(btns, text="All off", command=self.all_off).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )
        self.redraw()

    def _line_xs(self) -> np.ndarray:
        pad = 10.0
        inner = max(self.size - 2.0 * pad, 1.0)
        if self.resolution <= 1:
            return np.array([self.size / 2.0], dtype=np.float64)
        return pad + np.linspace(0.0, inner, self.resolution)

    def _nearest_index(self, x: float) -> int:
        xs = self._line_xs()
        idx = int(np.argmin(np.abs(xs - x)))
        spacing = float(xs[1] - xs[0]) if xs.size > 1 else 12.0
        if abs(xs[idx] - x) > max(6.0, 0.55 * spacing):
            return -1
        return idx

    def _on_click(self, axis: int, event) -> None:
        idx = self._nearest_index(event.x)
        if idx < 0:
            return
        self.enabled[axis][idx] = not bool(self.enabled[axis][idx])
        self.redraw()
        self.on_change()

    def _on_motion(self, axis: int, event) -> None:
        idx = self._nearest_index(event.x)
        hover = (axis, idx) if idx >= 0 else None
        if hover == self.hover:
            return
        self.hover = hover
        self.redraw()
        self.on_change()

    def _on_leave(self, axis: int) -> None:
        if self.hover is None or self.hover[0] != axis:
            return
        self.hover = None
        self.redraw()
        self.on_change()

    def all_on(self) -> None:
        for arr in self.enabled:
            arr.fill(True)
        self.redraw()
        self.on_change()

    def all_off(self) -> None:
        for arr in self.enabled:
            arr.fill(False)
        self.redraw()
        self.on_change()

    def redraw(self) -> None:
        xs = self._line_xs()
        y0, y1 = 8, self.size - 8
        for axis, canvas in enumerate(self._canvases):
            if axis >= self.n_visual:
                continue
            canvas.delete("line")
            on = self.enabled[axis]
            for i, x in enumerate(xs):
                hovered = self.hover == (axis, i)
                if hovered:
                    color = "#9ad1ff"
                    width = 3
                elif on[i]:
                    color = "#f0f0f0"
                    width = 2
                else:
                    color = "#4a4a4a"
                    width = 2
                canvas.create_line(
                    x, y0, x, y1, fill=color, width=width, tags="line"
                )


def _apply_layer(
    *,
    cube: NDCube,
    planets: list[Planet],
    planet_meshes,
    hit_markers,
    xyz: np.ndarray,
    extra_idx: np.ndarray,
    visual_idx: np.ndarray,
    hit: np.ndarray,
    slider_vals: list[int],
    planet_on: list[tk.BooleanVar],
    points_on: list[tk.BooleanVar],
    enabled_planes: list[np.ndarray],
    hover: tuple[int, int] | None,
    resolution: int,
    point_size: float,
    canvas,
    time: np.ndarray | None = None,
    t_max: float = T_MAX,
    color_by_time: bool = False,
) -> None:
    extra = _extra_coords(cube, resolution, slider_vals)
    layer = _layer_mask(extra_idx, slider_vals, resolution)
    union = _union_plane_mask(visual_idx, enabled_planes)
    on_hover = None
    if hover is not None:
        axis, hidx = hover
        if 0 <= axis < visual_idx.shape[1]:
            on_hover = visual_idx[:, axis] == int(hidx)
            vis = layer & (union | on_hover)
        else:
            vis = layer & union
    else:
        vis = layer & union

    for i, planet in enumerate(planets):
        ball = planet_3flat_ball(planet, cube, extra)
        fill, wire = planet_meshes[i]
        show_planet = bool(planet_on[i].get()) and ball is not None
        fill.visible = show_planet
        wire.visible = show_planet
        fill.mesh.visible = show_planet
        wire.mesh.visible = False
        wire.border.visible = show_planet
        if ball is not None:
            (x, y, z), rad = ball
            fill.transform.translate = (x, y, z)
            fill.transform.scale = (rad, rad, rad)
            outline = rad * 1.045
            wire.transform.translate = (x, y, z)
            wire.transform.scale = (outline, outline, outline)

        sel = vis & (hit == i)
        pts = xyz[sel]
        use_time = bool(color_by_time) and time is not None
        if pts.size == 0:
            pts = np.zeros((0, 3), dtype=np.float32)
            face = _rgb01(planet.color)
        elif use_time:
            rgb = time_to_hit_rgb(time[sel], t_max)
            r = rgb[:, 0].astype(np.float32) / 255.0
            g = rgb[:, 1].astype(np.float32) / 255.0
            b = rgb[:, 2].astype(np.float32) / 255.0
            if on_hover is not None:
                alphas = np.where(on_hover[sel], 1.0, HOVER_DIM_ALPHA).astype(np.float32)
            else:
                alphas = np.ones(pts.shape[0], dtype=np.float32)
            face = np.column_stack([r, g, b, alphas])
        elif on_hover is not None:
            r, g, b, _a = _rgb01(planet.color)
            alphas = np.where(on_hover[sel], 1.0, HOVER_DIM_ALPHA).astype(np.float32)
            face = np.column_stack(
                [
                    np.full(pts.shape[0], r, dtype=np.float32),
                    np.full(pts.shape[0], g, dtype=np.float32),
                    np.full(pts.shape[0], b, dtype=np.float32),
                    alphas,
                ]
            )
        else:
            face = _rgb01(planet.color)
        hit_markers[i].set_data(
            pts,
            face_color=face,
            edge_width=0,
            size=point_size,
            symbol="o",
        )
        hit_markers[i].visible = bool(points_on[i].get())
    canvas.update()


def _controls_window(
    canvas,
    planets,
    planet_meshes,
    hit_markers,
    hit: np.ndarray,
    positions: np.ndarray,
    xyz: np.ndarray,
    extra_idx: np.ndarray,
    visual_idx: np.ndarray,
    cube: NDCube,
    resolution: int,
    point_size: float,
    session: dict,
    time: np.ndarray | None = None,
    t_max: float = T_MAX,
) -> tk.Tk:
    root = tk.Tk()
    root.title("Basin controls")
    root.geometry("380x860")
    root.resizable(True, True)

    n_extra = extra_idx.shape[1]
    n_visual = visual_idx.shape[1]
    n_timeout = int(np.sum(hit < 0))
    ttk.Label(
        root,
        text=(
            f"r={resolution}   k={cube.k}   N={len(hit):,}\n"
            f"timeout (hidden)={n_timeout:,}   h={cube.half_extent:g}\n"
            "Drag: rotate   Scroll: zoom   Shift+drag: pan"
        ),
        justify=tk.LEFT,
    ).pack(anchor="w", padx=10, pady=(10, 8))
    view_status = ttk.Label(root, text="", justify=tk.LEFT)
    view_status.pack(anchor="w", padx=10, pady=(0, 6))

    planet_on: list[tk.BooleanVar] = []
    points_on: list[tk.BooleanVar] = []
    slider_vars: list[tk.IntVar] = []
    extra_labels: list[ttk.Label] = []
    picker: PlaneStripPicker | None = None
    color_by_time_var = tk.BooleanVar(value=False)

    def refresh(*_args):
        if len(planet_on) < len(planets) or len(points_on) < len(planets):
            return
        if picker is None:
            enabled = [np.ones(max(resolution, 1), dtype=bool) for _ in range(n_visual)]
            hover = None
        else:
            enabled = picker.enabled
            hover = picker.hover
        sliders = [_clamp_slider(v.get(), resolution) for v in slider_vars]
        extra = _extra_coords(cube, resolution, sliders)
        for j, lbl in enumerate(extra_labels):
            if j < len(extra):
                lbl.config(
                    text=f"u{j + 4}  {sliders[j]}/{resolution}   coord={extra[j]:+.4g}"
                )
        layer = _layer_mask(extra_idx, sliders, resolution)
        union = _union_plane_mask(visual_idx, enabled)
        n_slice = int((layer & (hit >= 0)).sum())
        n_show = int((layer & union & (hit >= 0)).sum())
        view_status.config(
            text=f"Showing {n_show:,} hit points  ({n_slice:,} on this extra slice)"
        )
        _apply_layer(
            cube=cube,
            planets=planets,
            planet_meshes=planet_meshes,
            hit_markers=hit_markers,
            xyz=xyz,
            extra_idx=extra_idx,
            visual_idx=visual_idx,
            hit=hit,
            slider_vals=sliders,
            planet_on=planet_on,
            points_on=points_on,
            enabled_planes=enabled,
            hover=hover,
            resolution=resolution,
            point_size=point_size,
            canvas=canvas,
            time=time,
            t_max=t_max,
            color_by_time=bool(color_by_time_var.get()),
        )

    if n_extra:
        slice_frame = ttk.LabelFrame(root, text="Extra dimensions (3-flat slice)")
        slice_frame.pack(fill="x", padx=10, pady=6)
        for j in range(n_extra):
            var = tk.IntVar(value=_mid_extra_slider(resolution, cube.half_extent))
            slider_vars.append(var)
            row = ttk.Frame(slice_frame)
            row.pack(fill="x", padx=8, pady=4)
            lbl = ttk.Label(row, text=f"u{j + 4}  1..{resolution}")
            lbl.pack(anchor="w")
            extra_labels.append(lbl)
            ttk.Scale(
                row,
                from_=1,
                to=max(resolution, 1),
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda _v, v=var: (
                    v.set(_clamp_slider(_v, resolution)),
                    refresh(),
                ),
            ).pack(fill="x")
            var.set(_mid_extra_slider(resolution, cube.half_extent))

    if n_visual:
        plane_frame = ttk.LabelFrame(root, text="Coordinate planes (union of slices)")
        plane_frame.pack(fill="x", padx=10, pady=6)
        picker = PlaneStripPicker(
            plane_frame,
            resolution=resolution,
            n_visual=n_visual,
            on_change=refresh,
        )
        picker.pack(fill="x", padx=6, pady=(4, 8))

    planet_frame = ttk.LabelFrame(root, text="Planets")
    planet_frame.pack(fill="x", padx=10, pady=6)
    for i, planet in enumerate(planets):
        var = tk.BooleanVar(value=True)
        planet_on.append(var)
        ttk.Checkbutton(
            planet_frame,
            text=f"Show planet {i + 1}",
            variable=var,
            command=refresh,
        ).pack(anchor="w", padx=8, pady=2)

    points_frame = ttk.LabelFrame(root, text="Hit points")
    points_frame.pack(fill="x", padx=10, pady=6)
    for i, planet in enumerate(planets):
        n_hits = int(np.sum(hit == i))
        var = tk.BooleanVar(value=True)
        points_on.append(var)
        ttk.Checkbutton(
            points_frame,
            text=f"Show planet {i + 1} points ({n_hits:,} total)",
            variable=var,
            command=refresh,
        ).pack(anchor="w", padx=8, pady=2)

    color_frame = ttk.LabelFrame(root, text="Coloring")
    color_frame.pack(fill="x", padx=10, pady=6)
    time_toggle = ttk.Checkbutton(
        color_frame,
        text="Color by time to hit  (red = fast, black = t_max)",
        variable=color_by_time_var,
        command=refresh,
    )
    time_toggle.pack(anchor="w", padx=8, pady=4)
    if time is None:
        time_toggle.state(["disabled"])

    io_frame = ttk.LabelFrame(root, text="Model")
    io_frame.pack(fill="x", padx=10, pady=8)

    def save_clicked():
        default_dir = Path(__file__).resolve().parent / "output"
        default_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=root,
            title="Save basin model",
            initialdir=str(default_dir),
            initialfile=f"basins_r{resolution}_k{cube.k}.npz",
            defaultextension=".npz",
            filetypes=FILE_TYPES,
        )
        if not path:
            return
        try:
            saved = save_model(
                path,
                positions=positions,
                hit=hit,
                time=time,
                planets=planets,
                cube=cube,
                resolution=resolution,
                point_size=point_size,
                extra={
                    "g": G,
                    "dt": DT,
                    "t_max": t_max,
                    "force_exponent": FORCE_EXPONENT,
                    "softening": SOFTENING,
                    "relativistic": RELATIVISTIC,
                    "c_light": C_LIGHT,
                },
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=root)
            return
        messagebox.showinfo("Saved", f"Model written to:\n{saved}", parent=root)

    def open_clicked():
        path = filedialog.askopenfilename(
            parent=root,
            title="Open basin model",
            initialdir=str(Path(__file__).resolve().parent / "output"),
            filetypes=FILE_TYPES,
        )
        if not path:
            return
        session["next_path"] = path
        canvas.close()
        root.destroy()

    ttk.Button(io_frame, text="Save model…", command=save_clicked).pack(
        fill="x", padx=8, pady=(8, 4)
    )
    ttk.Button(io_frame, text="Open model…", command=open_clicked).pack(
        fill="x", padx=8, pady=(0, 8)
    )

    def on_close():
        session["next_path"] = None
        canvas.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh()
    return root


def _simulate_new() -> dict:
    cube = NDCube(center=CUBE_CENTER, axes=CUBE_AXES, half_extent=HALF_EXTENT)
    planets = PLANETS
    print("Simulating n-D volume lattice (this can take a while)...")
    positions, hit, time = simulate_volume(
        planets,
        RESOLUTION,
        cube,
        g=G,
        dt=DT,
        t_max=T_MAX,
        force_exponent=FORCE_EXPONENT,
        softening=SOFTENING,
        relativistic=RELATIVISTIC,
        c_light=C_LIGHT,
    )
    return {
        "positions": positions,
        "hit": hit,
        "time": time,
        "t_max": T_MAX,
        "planets": planets,
        "cube": cube,
        "resolution": RESOLUTION,
        "point_size": POINT_SIZE,
        "path": None,
    }


def _run_session(model: dict) -> str | None:
    title = "Restricted n-body basins (3D)"
    if model.get("path"):
        title = f"{Path(model['path']).name} — n-body basins"
    canvas, planet_meshes, hit_markers, xyz, extra_idx, visual_idx, hit = _build_scene(
        model["positions"],
        model["hit"],
        model["planets"],
        model["cube"],
        model["resolution"],
        point_size=model["point_size"],
        title=title,
    )
    session = {"next_path": None}
    root = _controls_window(
        canvas,
        model["planets"],
        planet_meshes,
        hit_markers,
        hit,
        model["positions"],
        xyz,
        extra_idx,
        visual_idx,
        model["cube"],
        model["resolution"],
        model["point_size"],
        session,
        time=model.get("time"),
        t_max=float(model.get("t_max", T_MAX)),
    )

    def pump():
        canvas.app.process_events()
        if canvas._closed:
            try:
                root.destroy()
            except tk.TclError:
                pass
            return
        try:
            root.after(16, pump)
        except tk.TclError:
            pass

    root.after(16, pump)
    root.mainloop()
    return session.get("next_path")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else None
    while True:
        if path:
            print(f"Loading model {path}")
            model = load_model(path)
        else:
            model = _simulate_new()
        path = _run_session(model)
        if not path:
            break


if __name__ == "__main__":
    main()
