# Restrictive n-body problem

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-CuPy-00B5E2.svg)](https://cupy.dev/)
[![2D](https://img.shields.io/badge/2D-Manim-e07a5f.svg)](https://www.manim.community/)
[![3D](https://img.shields.io/badge/3D-VisPy-4c78a8.svg)](https://vispy.org/)

GPU-accelerated **basins of attraction** for a *restricted* n-body problem: planets are fixed, one test asteroid is launched from rest at each sample, and the pixel or lattice point is colored by which planet it hits.

<p align="center">
  <img src="docs/equilateral_basins_3840x2160.png" alt="2D slice of restricted three-body collision basins" width="900">
</p>

<p align="center"><em>2D slice through three equal masses on an equilateral triangle. Color = colliding planet. Black = timeout (no hit by <code>t_max</code>). White rings = planet cross-sections with the slice.</em></p>

## What it does

This is not a free n-body integration of all bodies. The planets stay put (positive mass attracts, negative mass would repel). Each asteroid feels Newtonian inverse-square gravity (`force_exponent = 3`), starts at rest, and is integrated with **per-particle RK4** on the GPU (CuPy). A hit is a true geometric collision with an n-sphere of radius `R`, including the segment swept during a step so trajectories cannot tunnel through a planet. Hits are **not** inferred from periapsis.

You can:

- Render a **2D still image**: one asteroid per pixel on a chosen 2-flat in n-D. They then move in the full ambient dimension `D`. White outlines mark where each n-sphere meets the image plane (`sqrt(R² − dist²)`; omitted if the sphere misses the plane).
- Sample a **k-dimensional cube** of initial conditions and inspect it in an interactive **3D viewer**. The white box is the first three cube axes. Extra axes (`u4…uk`) are chosen with sliders. Lattice slices parallel to `yz` / `xz` / `xy` can be toggled. Planet balls are the n-sphere ∩ current 3-flat.
- **Save / open** a 3D run as `.npz` so you can explore a long simulation without recomputing it.

Default planets are three equal masses on an equilateral triangle in the first two coordinates (extra coordinates `0`).

## Requirements

- Python **3.10+**
- An NVIDIA GPU with a **CuPy** build that matches your CUDA toolkit ([CuPy install](https://docs.cupy.dev/en/stable/install.html))
- `numpy`, `pillow`
- **2D stills:** [Manim](https://www.manim.community/)
- **3D viewer:** [VisPy](https://vispy.org/) with the **glfw** backend (and Tk for the control panel)

## Install

From the project root:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

```bash
pip install numpy pillow manim vispy glfw
pip install cupy-cuda12x
```

Replace `cupy-cuda12x` with the wheel for your CUDA version (`cupy-cuda11x`, `cupy-cuda12x`, …). See the CuPy install page if you are unsure.

## 2D images

Edit the class attributes at the top of [`n_body_problem.py`](n_body_problem.py) (`EquilateralBasins`):

| Knob | Meaning |
| --- | --- |
| `dimension` | Ambient dimension `D` (triangle sits in `xy`; extra coords are `0`) |
| `G`, `dt`, `t_max` | Gravity strength, max RK4 step, integration cutoff |
| `mass`, `radius`, `triangle_side`, `colors` | Planet properties |
| `plane_points` | Three n-D points: origin, image `+x`, image `+y`. `None` uses `view_center` in `xy` |
| `view_height` | Height of the framed rectangle on that plane (width follows image aspect) |

Render a still. Pixel count is the Manim resolution (`-r`); that is also the asteroid count (`width × height`):

```bash
manim -s -r 1920,1080 n_body_problem.py EquilateralBasins
manim -s -r 3840,2160 n_body_problem.py EquilateralBasins
```

The PNG is written to `output/equilateral_basins_{W}x{H}.png` (not Manim’s `media/` folder). Unhit pixels stay black. White rings are planet–plane cross-sections.

## 3D / n-D viewer

Edit the knobs at the top of [`viewer_3d.py`](viewer_3d.py) **before** starting a new simulation (they are ignored when you open a saved model):

| Knob | Meaning |
| --- | --- |
| `RESOLUTION` (`r`) | Lattice points per cube axis |
| `HALF_EXTENT` (`h`) | Cube coordinates run through `[-h, h]` on each axis |
| `CUBE_CENTER`, `CUBE_AXES` | Center `C` and `k` spanning vectors. First three axes are the visible box; further vectors add extra dimensions and sliders |
| `G`, `DT`, `T_MAX`, masses, radii | Same physics as the 2D scene |

Asteroid count is **`r^k`**. Example: `r = 48`, four axes → about **5.3 million** asteroids. That is slow and memory-heavy; drop `r` or `k` while experimenting.

```bash
python viewer_3d.py
python viewer_3d.py output/my_basins.npz
```

**Controls**

- **Drag** to rotate, **scroll** to zoom, **Shift+drag** to pan.
- **Extra dimensions:** each slider picks one lattice value on `u4…uk`, i.e. which 3-flat you are looking at. Default is the slice nearest extra-coordinate `0` (where the default triangle lives).
- **Coordinate planes:** three face diagrams, `r` lines each. A lattice point `(i, j, k)` lies on one `yz` plane (`i`), one `xz` plane (`j`), and one `xy` plane (`k`). Visible points are the **union** of enabled planes (after the extra-dim slice). Click a line to toggle; hover previews that plane; **All on** / **All off** reset the set.
- **Planets / hit points:** show or hide each planet ball and its colored markers. Planet meshes are the solid planet color with a white wire outline, sized to the apparent 3-ball `sqrt(R² − dist²)` (hidden if the n-sphere misses this 3-flat).
- **Save model / Open model:** `.npz` stores the **initial** lattice, hit indices, and cube metadata. Filters (sliders, planes) are view-only and are not saved.

Timeouts are not drawn. The log line `alive=… active=…` means: `alive` = not yet collided (including particles that will time out); `active` = still being stepped this iteration (alive and short of `t_max`). Collided asteroids are dropped from the GPU work set.

## Physics notes

- Force on an asteroid is `G m (p − x) / |p − x|^e` with default `e = 3` (inverse square). Softening avoids a singular `0/0` at a planet center.
- Step size is **CFL-limited** near a surface so a step cannot jump over a planet.
- Collision if the asteroid is inside radius `R` or the accepted segment intersects the sphere.
- Inf/NaN from a blown-up step is attributed to the nearest planet.

## License

[MIT](LICENSE) © 2026 Nirav Ramdhanie
