"""Manim still-image scene: a 2D slice of n-D restricted n-body basins.

Only asteroids that start on the chosen plane are simulated (one per pixel).
They then move in the full D-dimensional space. Pixel count comes from
``manim -r``.
"""

from __future__ import annotations

from pathlib import Path

from manim import ImageMobject, Scene, config

from simulation import (
    SlicePlane,
    equilateral_triangle_planets,
    render_basins,
    save_basins_png,
)


class EquilateralBasins(Scene):
    """Restricted 3-body basins on a plane through an equilateral triangle.

    Tune the class attributes below, then render a still with::

        manim -s -r 1920,1080 n_body_problem.py EquilateralBasins
        manim -s -r 3840,2160 n_body_problem.py EquilateralBasins
    """

    # Physics
    G = 1.0
    dt = 0.02
    t_max = 40.0
    force_exponent = 3.0
    softening = 1e-6

    # Planets live in this many dimensions (extra coords are 0 for the triangle).
    dimension = 3
    mass = 1.0
    radius = 0.05
    triangle_side = 1.0
    colors = (
        (255, 92, 87),
        (91, 192, 222),
        (132, 204, 107),
    )

    # Three n-D points define the plane: P0 origin, P1-P0 image +x, P2-P0 image +y.
    # view_height is how much of the plane is shown (width follows image aspect).
    view_height = 4
    plane_points: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    view_center = (0.0, 0.0)

    def construct(self):
        width = int(config.pixel_width)
        height = int(config.pixel_height)
        planets = equilateral_triangle_planets(
            side=self.triangle_side,
            mass=self.mass,
            radius=self.radius,
            colors=self.colors,
            dim=self.dimension,
        )
        plane = self._slice_plane(width, height)
        print(
            f"Slice plane origin={plane.origin}  "
            f"u={plane.axis_u}  v={plane.axis_v}  "
            f"size={plane.width:.3f}x{plane.height:.3f}"
        )

        rgb = render_basins(
            planets,
            width,
            height,
            plane=plane,
            g=self.G,
            dt=self.dt,
            t_max=self.t_max,
            force_exponent=self.force_exponent,
            softening=self.softening,
        )

        out_dir = Path(__file__).resolve().parent / "output"
        out = out_dir / f"equilateral_basins_{width}x{height}.png"
        save_basins_png(rgb, out)

        image = ImageMobject(str(out))
        image.set_height(config.frame_height)
        self.add(image)

    def _slice_plane(self, width: int, height: int) -> SlicePlane:
        aspect = width / height
        if self.plane_points is not None:
            p0, p1, p2 = self.plane_points
            return SlicePlane.from_points(
                p0,
                p1,
                p2,
                width=self.view_height * aspect,
                height=self.view_height,
                dim=self.dimension,
            )
        return SlicePlane.from_view_2d(
            self.view_center, self.view_height, aspect, dim=self.dimension
        )
