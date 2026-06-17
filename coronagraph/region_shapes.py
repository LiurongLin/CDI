from __future__ import annotations

import math


def normalize_region_shape(region_shape: str) -> str:
    value = str(region_shape).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "circle": "circle",
        "ring": "ring",
        "annulus": "ring",
        "ring_of_circle": "ring_of_circle",
        "ring_of_circles": "ring_of_circle",
    }
    if value not in aliases:
        raise ValueError("region_shape must be 'circle', 'ring', or 'ring_of_circle'.")
    return aliases[value]


def annulus_radii_from_width(
    mid_radius_lamD: float,
    width_lamD: float,
) -> tuple[float, float]:
    """
    Convert annulus mid-radius and width into inner/outer radii.

    For the new ``ring`` region shape, the annulus is centered on the optical
    axis and its midpoint radius matches the planet angular separation.
    """
    mid_radius = float(mid_radius_lamD)
    width = float(width_lamD)
    if mid_radius <= 0.0:
        raise ValueError("mid_radius_lamD must be > 0.")
    if width <= 0.0:
        raise ValueError("width_lamD must be > 0.")
    half_width = 0.5 * width
    if half_width >= mid_radius:
        raise ValueError("width_lamD must be < 2 * mid_radius_lamD for a valid annulus.")
    return (mid_radius - half_width, mid_radius + half_width)


def build_touching_circle_ring(
    requested_region_radius_lamD: float,
    orbit_radius_lamD: float,
    anchor_angle_rad: float,
    rotation_fraction: float = 0.0,
    min_circles: int = 3,
) -> dict:
    """
    Snap a requested circle radius to the nearest value that tiles a full ring.

    Circle centers lie on a ring of radius ``orbit_radius_lamD`` and neighboring
    circles touch each other. One circle center is anchored at
    ``anchor_angle_rad``.
    """
    requested_radius = float(requested_region_radius_lamD)
    orbit_radius = float(orbit_radius_lamD)
    anchor_angle = float(anchor_angle_rad)
    rotation_u = float(rotation_fraction)
    min_count = int(min_circles)

    if requested_radius <= 0.0:
        raise ValueError("requested_region_radius_lamD must be > 0.")
    if orbit_radius <= 0.0:
        raise ValueError("orbit_radius_lamD must be > 0 for ring regions.")
    if rotation_u < 0.0 or rotation_u > 1.0:
        raise ValueError("rotation_fraction must be within [0, 1].")
    if min_count < 3:
        raise ValueError("min_circles must be >= 3.")

    max_ratio = math.sin(math.pi / float(min_count))
    requested_ratio = requested_radius / orbit_radius
    if requested_ratio >= max_ratio:
        candidate_counts = [min_count]
    else:
        clipped_ratio = min(max(requested_ratio, 1e-12), max_ratio)
        n_est = math.pi / math.asin(clipped_ratio)
        n_floor = max(min_count, int(math.floor(n_est)))
        n_ceil = max(min_count, int(math.ceil(n_est)))
        candidate_counts = sorted(
            {
                min_count,
                n_floor - 1,
                n_floor,
                n_floor + 1,
                n_ceil,
                n_ceil + 1,
            }
        )
        candidate_counts = [n for n in candidate_counts if n >= min_count]

    best_count = min(
        candidate_counts,
        key=lambda n: (
            abs(orbit_radius * math.sin(math.pi / float(n)) - requested_radius),
            n,
        ),
    )
    resolved_radius = orbit_radius * math.sin(math.pi / float(best_count))
    dtheta = 2.0 * math.pi / float(best_count)
    edge_cut_rotation_rad = 2.0 * math.asin(
        min(max(resolved_radius / max(2.0 * orbit_radius, 1e-20), 0.0), 1.0)
    )
    applied_rotation_rad = rotation_u * edge_cut_rotation_rad
    centers = [
        (
            float(orbit_radius * math.cos(anchor_angle + applied_rotation_rad + k * dtheta)),
            float(orbit_radius * math.sin(anchor_angle + applied_rotation_rad + k * dtheta)),
        )
        for k in range(best_count)
    ]

    return {
        "requested_radius_lamD": requested_radius,
        "resolved_radius_lamD": float(resolved_radius),
        "orbit_radius_lamD": orbit_radius,
        "anchor_angle_rad": anchor_angle,
        "rotation_fraction": rotation_u,
        "edge_cut_rotation_rad": float(edge_cut_rotation_rad),
        "applied_rotation_rad": float(applied_rotation_rad),
        "n_circles": int(best_count),
        "centers_lamD": centers,
        "center_angle_step_rad": float(dtheta),
    }
