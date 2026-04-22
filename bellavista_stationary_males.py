#!/usr/bin/env python3
"""Encounter-time simulation in Bellavista with stationary males.

This script loads the Bellavista boundary from the uploaded GeoJSON, projects it to
local Cartesian coordinates (meters), places one stationary male mosquito per square
meter of polygon area (rounded to the nearest integer), and simulates a single female
as a reflecting Brownian particle until she comes within perceptual radius R of any male.

It also reports a leading-order correction to estimate the mean encounter time when
males are not stationary but instead diffuse independently with diffusivity D_m:

    E[T_true] \approx E[T_stationary] * D_f / (D_f + D_m)

This uses the relative-diffusivity scaling D_rel = D_f + D_m.
"""

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
from scipy.spatial import cKDTree

# -----------------------------------------------------------------------------
# Projection / geometry helpers
# -----------------------------------------------------------------------------

LAT0 = -0.6958902314708109
LON0 = -90.32546837031445
DEFAULT_GEOJSON = Path("bellavista.json")


def make_local_transformer(lon0: float = LON0, lat0: float = LAT0) -> Transformer:
    """Return WGS84 -> local azimuthal-equidistant transformer in meters."""
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)


def polygon_area(vertices: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_closed(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        raise ValueError("Empty coordinate list.")
    if np.allclose(coords[0], coords[-1]):
        return coords
    return np.vstack([coords, coords[0]])


def segment_lengths(vertices: np.ndarray) -> np.ndarray:
    nxt = np.roll(vertices, -1, axis=0)
    return np.linalg.norm(nxt - vertices, axis=1)


@dataclass
class PolygonDomain:
    vertices: np.ndarray  # shape (n, 2), unclosed vertex list
    path: MplPath
    area: float
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    edge_lengths: np.ndarray

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)


def load_bellavista_domain(geojson_path: Path | str = DEFAULT_GEOJSON) -> PolygonDomain:
    """Load Bellavista LineString/Polygon and project to local Cartesian coordinates."""
    geojson_path = Path(geojson_path)
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        raise ValueError("GeoJSON contains no features.")

    geom = features[0]["geometry"]
    gtype = geom["type"]
    if gtype == "LineString":
        lonlat = np.asarray(geom["coordinates"], dtype=float)
    elif gtype == "Polygon":
        lonlat = np.asarray(geom["coordinates"][0], dtype=float)
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")

    transformer = make_local_transformer()
    x, y = transformer.transform(lonlat[:, 0], lonlat[:, 1])
    xy = np.column_stack([x, y])
    xy_closed = ensure_closed(xy)
    vertices = xy_closed[:-1].copy()

    area = polygon_area(vertices)
    path = MplPath(xy_closed)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    edges = segment_lengths(vertices)

    return PolygonDomain(
        vertices=vertices,
        path=path,
        area=area,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        edge_lengths=edges,
    )


# -----------------------------------------------------------------------------
# Sampling / reflection helpers
# -----------------------------------------------------------------------------


def sample_points_in_polygon(domain: PolygonDomain, n: int, rng: np.random.Generator) -> np.ndarray:
    """Rejection-sample n points uniformly from the polygon."""
    pts = np.empty((n, 2), dtype=float)
    filled = 0
    while filled < n:
        need = n - filled
        # modest oversampling factor
        batch = max(1000, int(math.ceil(need * 1.6)))
        trial = rng.uniform(domain.bbox_min, domain.bbox_max, size=(batch, 2))
        mask = domain.path.contains_points(trial)
        accepted = trial[mask]
        take = min(need, len(accepted))
        if take > 0:
            pts[filled : filled + take] = accepted[:take]
            filled += take
    return pts


def reflect_step_single(pos: np.ndarray, step: np.ndarray, domain: PolygonDomain, max_bounces: int = 16) -> np.ndarray:
    """Advance one point by step with specular reflection at polygon edges.

    This is an approximate but practical polygon reflection routine suitable for small dt.
    """
    p = pos.astype(float).copy()
    v = step.astype(float).copy()
    verts = domain.vertices
    n = len(verts)

    for _ in range(max_bounces):
        candidate = p + v
        if domain.path.contains_point(candidate):
            return candidate

        best_t = None
        best_normal = None
        best_edge_idx = None

        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            e = b - a
            denom = v[0] * e[1] - v[1] * e[0]
            if abs(denom) < 1e-14:
                continue
            ap = a - p
            t = (ap[0] * e[1] - ap[1] * e[0]) / denom
            u = (ap[0] * v[1] - ap[1] * v[0]) / denom
            if 1e-12 <= t <= 1.0 + 1e-12 and -1e-12 <= u <= 1.0 + 1e-12:
                if best_t is None or t < best_t:
                    best_t = max(0.0, min(1.0, t))
                    tangent = e / np.linalg.norm(e)
                    normal = np.array([-tangent[1], tangent[0]])
                    mid = 0.5 * (a + b)
                    if np.dot((domain.vertices.mean(axis=0) - mid), normal) < 0:
                        normal = -normal
                    best_normal = normal
                    best_edge_idx = i

        if best_t is None:
            # Fallback: reject step if intersection not resolved.
            return p

        p = p + best_t * v
        remaining = (1.0 - best_t) * v
        remaining = remaining - 2.0 * np.dot(remaining, best_normal) * best_normal
        p = p + 1e-10 * best_normal
        v = remaining

    return p


# -----------------------------------------------------------------------------
# Simulation helpers
# -----------------------------------------------------------------------------


def recommended_dt(D_female: float, R: float, safety: float = 0.1) -> float:
    """Choose dt so RMS step length is safety * R.

    In 2D Brownian motion with diffusivity D, one coordinate increment has variance 2D dt,
    so RMS displacement magnitude is sqrt(4 D dt). Setting that equal to safety * R gives

        dt = (safety * R)^2 / (4 D)
    """
    if D_female <= 0:
        raise ValueError("D_female must be positive.")
    if R <= 0:
        raise ValueError("R must be positive.")
    return (safety * R) ** 2 / (4.0 * D_female)


@dataclass
class StationarySimulationResult:
    encounter_time: float
    encountered_at_start: bool
    female_start: np.ndarray
    encounter_position: np.ndarray
    encountered_male_index: int
    male_count: int
    area_m2: float
    dt: float
    steps: int
    female_path: Optional[np.ndarray] = None
    males: Optional[np.ndarray] = None


@dataclass
class MeanEncounterEstimate:
    n_runs: int
    sample_mean_stationary: float
    sample_std_stationary: float
    ci95_stationary: tuple[float, float]
    half_width_stationary: float
    sample_mean_corrected: float
    ci95_corrected: tuple[float, float]
    correction_factor: float
    D_female: float
    D_male: float



def first_encounter_time_stationary_males(
    domain: PolygonDomain,
    D_female: float,
    R: float,
    *,
    male_count: Optional[int] = None,
    dt: Optional[float] = None,
    max_time: float = 1e6,
    rng: Optional[np.random.Generator] = None,
    return_female_path: bool = False,
    return_males: bool = False,
) -> StationarySimulationResult:
    """Simulate one moving female among stationary males until first encounter."""
    if rng is None:
        rng = np.random.default_rng()
    if male_count is None:
        male_count = int(round(domain.area))
    if male_count <= 0:
        raise ValueError("male_count must be positive.")
    if dt is None:
        dt = recommended_dt(D_female, R)
        #print(f"Using recommended time step dt = {dt}")

    males = sample_points_in_polygon(domain, male_count, rng)
    female = sample_points_in_polygon(domain, 1, rng)[0]
    female_start = female.copy()

    tree = cKDTree(males)
    nearby = tree.query_ball_point(female, r=R)
    if nearby:
        fem_path = np.array([female.copy()]) if return_female_path else None
        return StationarySimulationResult(
            encounter_time=0.0,
            encountered_at_start=True,
            female_start=female_start,
            encounter_position=female.copy(),
            encountered_male_index=int(nearby[0]),
            male_count=male_count,
            area_m2=domain.area,
            dt=dt,
            steps=0,
            female_path=fem_path,
            males=males if return_males else None,
        )

    sigma = math.sqrt(2.0 * D_female * dt)
    path = [female.copy()] if return_female_path else None
    max_steps = int(math.ceil(max_time / dt))

    for step_idx in range(1, max_steps + 1):
        step = sigma * rng.standard_normal(size=2)
        female = reflect_step_single(female, step, domain)
        if return_female_path:
            path.append(female.copy())

        nearby = tree.query_ball_point(female, r=R)
        if nearby:
            return StationarySimulationResult(
                encounter_time=step_idx * dt,
                encountered_at_start=False,
                female_start=female_start,
                encounter_position=female.copy(),
                encountered_male_index=int(nearby[0]),
                male_count=male_count,
                area_m2=domain.area,
                dt=dt,
                steps=step_idx,
                female_path=np.array(path) if return_female_path else None,
                males=males if return_males else None,
            )

    raise RuntimeError("No encounter before max_time; increase max_time or inspect parameters.")



def correction_factor_stationary_to_moving(D_female: float, D_male: float) -> float:
    """Leading-order mean-time correction: T_true ~= T_stationary * Df / (Df + Dm)."""
    if D_female <= 0 or D_male < 0:
        raise ValueError("Require D_female > 0 and D_male >= 0.")
    return D_female / (D_female + D_male)



def mean_first_encounter_time_stationary_males(
    domain: PolygonDomain,
    D_female: float,
    R: float,
    *,
    D_male_for_correction: float = 1080,
    N: int = 100,
    male_count: Optional[int] = None,
    dt: Optional[float] = None,
    max_time: float = 1e6,
    seed: Optional[int] = None,
    progress_every: int = 10,
) -> MeanEncounterEstimate:
    """Monte Carlo estimate for stationary males and corrected moving-males mean."""
    rng = np.random.default_rng(seed)
    times = np.empty(N, dtype=float)

    for i in range(N):
        res = first_encounter_time_stationary_males(
            domain,
            D_female,
            R,
            male_count=male_count,
            dt=dt,
            max_time=max_time,
            rng=rng,
            return_female_path=False,
            return_males=False,
        )
        times[i] = res.encounter_time
        if progress_every and ((i + 1) % progress_every == 0 or i + 1 == N):
            m = times[: i + 1].mean()
            s = times[: i + 1].std(ddof=1) if i > 0 else 0.0
            print(f"Simulation {i+1} of {N}. Running mean/std: {m}/{s}")

    mean = float(times.mean())
    std = float(times.std(ddof=1)) if N > 1 else 0.0
    tcrit = 1.96 if N > 50 else 2.0  # mild simplification; good enough for reporting
    half = tcrit * std / math.sqrt(N) if N > 1 else 0.0
    ci = (mean - half, mean + half)

    factor = correction_factor_stationary_to_moving(D_female, D_male_for_correction)
    mean_corr = factor * mean
    ci_corr = (factor * ci[0], factor * ci[1])

    return MeanEncounterEstimate(
        n_runs=N,
        sample_mean_stationary=mean,
        sample_std_stationary=std,
        ci95_stationary=ci,
        half_width_stationary=half,
        sample_mean_corrected=mean_corr,
        ci95_corrected=ci_corr,
        correction_factor=factor,
        D_female=D_female,
        D_male=D_male_for_correction,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_stationary_simulation(
    domain: PolygonDomain,
    result: StationarySimulationResult,
    *,
    male_plot_cap: int = 1000,
    show_males: bool = True,
    ax=None,
):
    """Plot polygon, female path, and optionally a subset of stationary males."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    closed = np.vstack([domain.vertices, domain.vertices[0]])
    ax.plot(closed[:, 0], closed[:, 1], linewidth=1.5, label="Bellavista boundary")

    if show_males and result.males is not None:
        males = result.males
        if len(males) > male_plot_cap:
            idx = np.random.default_rng(0).choice(len(males), size=male_plot_cap, replace=False)
            males = males[idx]
        ax.scatter(males[:, 0], males[:, 1], s=2, alpha=0.25, label="Stationary males (sample)")

    if result.female_path is not None:
        fp = result.female_path
        ax.plot(fp[:, 0], fp[:, 1], linewidth=1.2, label="Female path")
        ax.scatter(fp[0, 0], fp[0, 1], s=40, marker="o", label="Female start")

    ax.scatter(
        result.encounter_position[0],
        result.encounter_position[1],
        s=60,
        marker="*",
        label="Encounter",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Stationary-male encounter in Bellavista\n"
        f"time={result.encounter_time:.3f}, males={result.male_count}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------


def _demo():
    domain = load_bellavista_domain(DEFAULT_GEOJSON)
    print(f"Bellavista area: {domain.area:.2f} m^2")
    print(f"Rounded male count: {round(domain.area)}")

    D_female = 1080.0
    R = 0.199
    D_male = 1080.0

    res = first_encounter_time_stationary_males(
        domain,
        D_female=D_female,
        R=R,
        return_female_path=True,
        return_males=True
    )
    print(f"Single-run stationary encounter time: {res.encounter_time:.6f}")

    est = mean_first_encounter_time_stationary_males(
        domain,
        D_female=D_female,
        R=R,
        D_male_for_correction=D_male,
        N=200,
        progress_every=10,
    )
    print(est)

    # plot_stationary_simulation(domain, res)
    # plt.show()


if __name__ == "__main__":
    _demo()
