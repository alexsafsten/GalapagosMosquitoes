from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from pyproj import CRS, Transformer


# -----------------------------------------------------------------------------
# Projection and polygon loading
# -----------------------------------------------------------------------------

ORIGIN_LAT = -0.6958902314708109
ORIGIN_LON = -90.32546837031445


def make_local_transformer(
    origin_lat: float = ORIGIN_LAT,
    origin_lon: float = ORIGIN_LON,
) -> Transformer:
    """Create a local WGS84 azimuthal-equidistant projection in meters."""
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )
    wgs84 = CRS.from_epsg(4326)
    return Transformer.from_crs(wgs84, local_crs, always_xy=True)


def _close_ring(coords: np.ndarray) -> np.ndarray:
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    return coords


def load_bellavista_polygon(
    json_path: str | Path = "bellavista.json",
    origin_lat: float = ORIGIN_LAT,
    origin_lon: float = ORIGIN_LON,
) -> np.ndarray:
    """Load Bellavista boundary from GeoJSON and project to local Cartesian coords.

    Returns
    -------
    polygon : (n+1, 2) ndarray
        Closed polygon vertices in meters.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    geom = data["features"][0]["geometry"]
    coords_ll = np.asarray(geom["coordinates"], dtype=float)
    transformer = make_local_transformer(origin_lat=origin_lat, origin_lon=origin_lon)
    x, y = transformer.transform(coords_ll[:, 0], coords_ll[:, 1])
    polygon = np.column_stack([x, y])
    return _close_ring(polygon)


def polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area in square meters for a closed polygon."""
    poly = _close_ring(np.asarray(polygon, dtype=float))
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def polygon_edges(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poly = _close_ring(np.asarray(polygon, dtype=float))
    return poly[:-1].copy(), poly[1:].copy()


@dataclass
class PolygonDomain:
    polygon: np.ndarray

    def __post_init__(self) -> None:
        self.polygon = _close_ring(np.asarray(self.polygon, dtype=float))
        self.path = MplPath(self.polygon)
        self.edge_starts, self.edge_ends = polygon_edges(self.polygon)
        self.edge_vecs = self.edge_ends - self.edge_starts
        self.edge_lens2 = np.sum(self.edge_vecs**2, axis=1)
        self.centroid = self._compute_centroid()
        self._bbox_min = self.polygon[:-1].min(axis=0)
        self._bbox_max = self.polygon[:-1].max(axis=0)

    def _compute_centroid(self) -> np.ndarray:
        x = self.polygon[:, 0]
        y = self.polygon[:, 1]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        area2 = np.sum(cross)
        if abs(area2) < 1e-14:
            return np.mean(self.polygon[:-1], axis=0)
        cx = np.sum((x[:-1] + x[1:]) * cross) / (3.0 * area2)
        cy = np.sum((y[:-1] + y[1:]) * cross) / (3.0 * area2)
        return np.array([cx, cy], dtype=float)

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def bbox_min(self) -> np.ndarray:
        return self._bbox_min

    @property
    def bbox_max(self) -> np.ndarray:
        return self._bbox_max

    def contains_points(self, pts: np.ndarray) -> np.ndarray:
        return self.path.contains_points(np.asarray(pts, dtype=float), radius=1e-12)

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Rejection-sample n approximately uniform points in polygon."""
        pts = np.empty((n, 2), dtype=float)
        filled = 0
        width = self.bbox_max - self.bbox_min
        while filled < n:
            need = n - filled
            batch = max(4 * need, 10000)
            trial = self.bbox_min + rng.random((batch, 2)) * width
            mask = self.contains_points(trial)
            accepted = trial[mask]
            take = min(need, len(accepted))
            if take > 0:
                pts[filled : filled + take] = accepted[:take]
                filled += take
        return pts

    def _nearest_edge_data(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each point, find nearest point on polygon boundary and edge tangent."""
        pts = np.asarray(pts, dtype=float)
        # Shapes: (m, e, 2)
        ap = pts[:, None, :] - self.edge_starts[None, :, :]
        t = np.sum(ap * self.edge_vecs[None, :, :], axis=2) / self.edge_lens2[None, :]
        t = np.clip(t, 0.0, 1.0)
        projections = self.edge_starts[None, :, :] + t[:, :, None] * self.edge_vecs[None, :, :]
        diff = pts[:, None, :] - projections
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(dist2, axis=1)
        nearest = projections[np.arange(len(pts)), idx]
        tangents = self.edge_vecs[idx]
        tangent_norm = np.linalg.norm(tangents, axis=1)
        tangent_norm = np.where(tangent_norm > 0.0, tangent_norm, 1.0)
        tangents = tangents / tangent_norm[:, None]
        return nearest, tangents, idx

    def reflect_points(self, prev: np.ndarray, proposed: np.ndarray, max_iter: int = 3) -> np.ndarray:
        """Reflect boundary crossings back into polygon.

        This is an approximate specular reflection that is reliable when dt is small.
        """
        prev = np.asarray(prev, dtype=float)
        curr = np.asarray(proposed, dtype=float).copy()

        for _ in range(max_iter):
            inside = self.contains_points(curr)
            if np.all(inside):
                return curr
            bad = ~inside
            pts = curr[bad]
            nearest, tangents, _ = self._nearest_edge_data(pts)
            disp = pts - nearest
            normal_comp = disp - np.sum(disp * tangents, axis=1)[:, None] * tangents
            reflected = pts - 2.0 * normal_comp
            curr[bad] = reflected

        # Final fallback: move bad points to a nearby interior point along boundary->centroid.
        inside = self.contains_points(curr)
        if np.all(inside):
            return curr
        bad = ~inside
        pts = curr[bad]
        nearest, _, _ = self._nearest_edge_data(pts)
        direction = self.centroid[None, :] - nearest
        norms = np.linalg.norm(direction, axis=1)
        norms = np.where(norms > 0.0, norms, 1.0)
        direction = direction / norms[:, None]
        curr[bad] = nearest + 1e-8 * direction

        # If a point is still outside due to numerical weirdness, keep previous position.
        inside = self.contains_points(curr)
        if not np.all(inside):
            curr[~inside] = prev[~inside]
        return curr


# -----------------------------------------------------------------------------
# Simulation helpers
# -----------------------------------------------------------------------------


def recommended_dt(
    perceptual_radius: float,
    female_diffusion: float,
    male_diffusion: float,
    safety: float = 10.0,
) -> float:
    """Choose dt so RMS step length is much smaller than the radius.

    For 2D Brownian motion with diffusion D, RMS step length is sqrt(4 D dt).
    We require the larger of the male/female RMS step lengths to be <= R / safety.
    """
    max_d = max(float(female_diffusion), float(male_diffusion))
    if max_d <= 0.0:
        return 1.0
    return perceptual_radius**2 / (4.0 * max_d * safety**2)


@dataclass
class ManyMaleSimulationResult:
    encounter_time: float
    female_start: np.ndarray
    encounter_index: int
    encounter_position: np.ndarray
    area_m2: float
    n_males: int
    dt: float
    female_path: Optional[np.ndarray] = None
    male_positions_at_encounter: Optional[np.ndarray] = None
    male_sample_paths: Optional[np.ndarray] = None



def first_encounter_time_many_males(
    domain: PolygonDomain,
    perceptual_radius: float,
    female_diffusion: float,
    male_diffusion: float,
    n_males: Optional[int] = None,
    dt: Optional[float] = None,
    max_time: float = 1.0e6,
    seed: Optional[int] = None,
    return_female_path: bool = False,
    return_male_positions_at_encounter: bool = False,
    tracked_male_count: int = 0,
) -> ManyMaleSimulationResult:
    """Simulate one female and many moving males in Bellavista until encounter.

    Parameters
    ----------
    n_males : int, optional
        Defaults to round(area of polygon in square meters).
    tracked_male_count : int
        Number of male trajectories to record for plotting/debugging. Recording all
        male paths is intentionally unsupported because it is too memory-intensive.
    """
    rng = np.random.default_rng(seed)
    area_m2 = domain.area
    if n_males is None:
        n_males = int(round(area_m2))
    if n_males <= 0:
        raise ValueError("n_males must be positive.")
    if perceptual_radius <= 0.0:
        raise ValueError("perceptual_radius must be positive.")
    if female_diffusion < 0.0 or male_diffusion < 0.0:
        raise ValueError("diffusion coefficients must be nonnegative.")

    if dt is None:
        dt = recommended_dt(perceptual_radius, female_diffusion, male_diffusion)

    female = domain.sample_uniform(1, rng)[0]
    males = domain.sample_uniform(n_males, rng)

    # Immediate encounter check.
    dx0 = males[:, 0] - female[0]
    dy0 = males[:, 1] - female[1]
    dist2 = dx0 * dx0 + dy0 * dy0
    hit_mask = dist2 < perceptual_radius**2
    female_path = [female.copy()] if return_female_path else None

    tracked_ids = None
    male_sample_paths = None
    if tracked_male_count > 0:
        tracked_male_count = min(tracked_male_count, n_males)
        tracked_ids = rng.choice(n_males, size=tracked_male_count, replace=False)
        male_sample_paths = [males[tracked_ids].copy()]

    if np.any(hit_mask):
        idx = int(np.flatnonzero(hit_mask)[0])
        return ManyMaleSimulationResult(
            encounter_time=0.0,
            female_start=female.copy(),
            encounter_index=idx,
            encounter_position=males[idx].copy(),
            area_m2=area_m2,
            n_males=n_males,
            dt=dt,
            female_path=np.asarray(female_path) if female_path is not None else None,
            male_positions_at_encounter=males.copy() if return_male_positions_at_encounter else None,
            male_sample_paths=np.asarray(male_sample_paths) if male_sample_paths is not None else None,
        )

    sigma_f = math.sqrt(2.0 * female_diffusion * dt)
    sigma_m = math.sqrt(2.0 * male_diffusion * dt)
    n_steps = int(math.ceil(max_time / dt))

    for step in range(1, n_steps + 1):
        print(f"t = {step * dt}")
        female_prev = female[None, :].copy()
        female_prop = female_prev + sigma_f * rng.standard_normal((1, 2))
        female = domain.reflect_points(female_prev, female_prop)[0]

        males_prev = males.copy()
        males_prop = males_prev + sigma_m * rng.standard_normal((n_males, 2))
        males = domain.reflect_points(males_prev, males_prop)

        if female_path is not None:
            female_path.append(female.copy())
        if male_sample_paths is not None and tracked_ids is not None:
            male_sample_paths.append(males[tracked_ids].copy())

        dx = males[:, 0] - female[0]
        dy = males[:, 1] - female[1]
        dist2 = dx * dx + dy * dy
        hit_mask = dist2 < perceptual_radius**2
        if np.any(hit_mask):
            idx = int(np.flatnonzero(hit_mask)[0])
            return ManyMaleSimulationResult(
                encounter_time=step * dt,
                female_start=female_path[0].copy() if female_path is not None else female.copy(),
                encounter_index=idx,
                encounter_position=males[idx].copy(),
                area_m2=area_m2,
                n_males=n_males,
                dt=dt,
                female_path=np.asarray(female_path) if female_path is not None else None,
                male_positions_at_encounter=(males.copy() if return_male_positions_at_encounter else None),
                male_sample_paths=(np.asarray(male_sample_paths) if male_sample_paths is not None else None),
            )

    raise RuntimeError(
        "No encounter occurred before max_time. Increase max_time or inspect parameters."
    )



def mean_first_encounter_time_many_males(
    domain: PolygonDomain,
    perceptual_radius: float,
    female_diffusion: float,
    male_diffusion: float,
    n_trials: int,
    n_males: Optional[int] = None,
    dt: Optional[float] = None,
    max_time: float = 1.0e6,
    seed: Optional[int] = None,
    progress_every: int = 10,
) -> dict:
    """Monte Carlo estimate of the mean encounter time for one female and many males."""
    rng = np.random.default_rng(seed)
    times = np.empty(n_trials, dtype=float)

    for k in range(n_trials):
        res = first_encounter_time_many_males(
            domain=domain,
            perceptual_radius=perceptual_radius,
            female_diffusion=female_diffusion,
            male_diffusion=male_diffusion,
            n_males=n_males,
            dt=dt,
            max_time=max_time,
            seed=int(rng.integers(0, 2**63 - 1)),
            return_female_path=False,
            return_male_positions_at_encounter=False,
            tracked_male_count=0,
        )
        times[k] = res.encounter_time
        if progress_every > 0 and ((k + 1) % progress_every == 0 or k + 1 == n_trials):
            mean = np.mean(times[: k + 1])
            std = np.std(times[: k + 1], ddof=1) if k >= 1 else float("nan")
            print(f"Trial {k+1} of {n_trials}. Running mean/std: {mean}/{std}")

    mean = float(np.mean(times))
    std = float(np.std(times, ddof=1)) if n_trials > 1 else float("nan")
    half_width = float(1.96 * std / math.sqrt(n_trials)) if n_trials > 1 else float("nan")
    return {
        "times": times,
        "mean": mean,
        "std": std,
        "ci95": (mean - half_width, mean + half_width) if n_trials > 1 else (float("nan"), float("nan")),
        "relative_half_width": (half_width / mean) if n_trials > 1 and mean > 0 else float("nan"),
        "n_trials": n_trials,
        "n_males": int(round(domain.area)) if n_males is None else n_males,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_many_male_simulation(
    domain: PolygonDomain,
    result: ManyMaleSimulationResult,
    show_all_males_at_encounter: bool = True,
    max_males_to_plot: int = 25000,
    figsize: tuple[float, float] = (8.0, 8.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Bellavista polygon, female path, and optional male points/paths.

    By default this will subsample the male cloud if it is large.
    """
    fig, ax = plt.subplots(figsize=figsize)
    poly = domain.polygon
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=1.5, label="Bellavista boundary")

    if result.female_path is not None:
        ax.plot(result.female_path[:, 0], result.female_path[:, 1], lw=2.0, label="Female path")
        ax.plot(result.female_path[0, 0], result.female_path[0, 1], "go", ms=6, label="Female start")
        ax.plot(result.female_path[-1, 0], result.female_path[-1, 1], "ro", ms=6, label="Encounter point")

    if result.male_sample_paths is not None:
        sample_paths = np.asarray(result.male_sample_paths)
        for j in range(sample_paths.shape[1]):
            ax.plot(sample_paths[:, j, 0], sample_paths[:, j, 1], lw=1.0, alpha=0.7)

    if show_all_males_at_encounter and result.male_positions_at_encounter is not None:
        pts = result.male_positions_at_encounter
        if len(pts) > max_males_to_plot:
            idx = np.random.default_rng(0).choice(len(pts), size=max_males_to_plot, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.15, label="Male positions")

    ax.add_patch(
        plt.Circle(
            result.encounter_position,
            0.0,
            fill=False,
        )
    )
    ax.scatter(
        [result.encounter_position[0]],
        [result.encounter_position[1]],
        c="red",
        s=20,
        zorder=5,
        label="Encountering male",
    )
    if result.female_path is not None:
        R = np.linalg.norm(result.female_path[-1] - result.encounter_position)
        circ = plt.Circle(result.female_path[-1], R, fill=False, ls="--", lw=1.0)
        ax.add_patch(circ)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"One female, {result.n_males:,} males\nEncounter time = {result.encounter_time:.4g}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------


def build_default_domain(json_path: str | Path = "bellavista.json") -> PolygonDomain:
    return PolygonDomain(load_bellavista_polygon(json_path))


if __name__ == "__main__":
    domain = build_default_domain("bellavista.json")
    print(f"Bellavista area: {domain.area:,.2f} m^2")
    print(f"Default male count (rounded area): {int(round(domain.area)):,}")

    result = first_encounter_time_many_males(
        domain=domain,
        perceptual_radius=0.199,
        female_diffusion=1080.0,
        male_diffusion=1080.0,
        n_males=None,
        dt=None,
        max_time=5_000.0,
        seed=12345,
        return_female_path=True,
        return_male_positions_at_encounter=True,
        tracked_male_count=20,
    )
    print(f"Encounter time: {result.encounter_time:.6f}")

    # fig, ax = plot_many_male_simulation(domain, result)
    # plt.show()
