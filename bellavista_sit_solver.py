#!/usr/bin/env python3
"""
bellavista_sit_solver.py

Solve the Bellavista SIT system

    dE/dt = phi(t) F (1 - E / k(t, x)) - (sigma_E(t) + mu_E(t)) E

    dF/dt - d_F(t) Delta F
        = ((E / (gamma(t) + zeta(t) (E + eta M))) - mu_F(t)) F

    dM/dt - d_M(t) Delta M
        = sum_{i=1}^K S_i delta(x) delta(t - i tau) - mu_M(t) M

on a polygonal mesh of Bellavista with homogeneous Neumann boundary conditions.

This first implementation is intentionally practical and easy to modify:

- The mesh is generated using the helper functions in ``bellavista_mesh.py``.
- The PDE discretization uses linear P1 finite elements via scikit-fem.
- Time stepping is done with a simple first-order IMEX / backward Euler scheme.
- The sterile-male impulses are handled as instantaneous jumps at the mesh node
  nearest the origin, which should be the embedded point (0, 0) if the mesh was
  generated with the updated mesh builder.

The code is written so that every model parameter is a Python callable.  For
now, those callables return constants, but you can later replace them by any
functions of time (and, for k, of time and space).

Important note
--------------
This is a good prototype solver, but not yet a production-quality biological or
numerical model.  In particular:

1. The point impulse for M is applied at a single nodal degree of freedom.
2. E is advanced nodewise using an explicit Euler step.
3. F is advanced with implicit diffusion and a nodal reaction coefficient frozen
   from the previous time step.
4. k(t, x) is evaluated at nodal coordinates.

These choices are reasonable for a first working code and are easy to upgrade
later.

Dependencies
------------
- numpy
- scipy
- matplotlib
- scikit-fem
- pyproj
- gmsh

Example
-------
    python bellavista_sit_solver.py bellavista.json --h 10 --dt 0.1 --show-mesh --show-results
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu
from skfem import Basis, MeshTri
from skfem.assembly import BilinearForm, asm
from skfem.element import ElementTriP1
from skfem.helpers import dot, grad
from matplotlib.widgets import Slider, RadioButtons

# Reuse the geometry / meshing code already written for Bellavista.
from bellavista_mesh import build_polygon_mesh, latlon_to_cartesian, plot_mesh

# Global references so matplotlib widgets do not get garbage collected
_time_slider = None
_field_radio = None


# -----------------------------------------------------------------------------
# Type aliases for clarity.
# -----------------------------------------------------------------------------
ScalarTimeFunc = Callable[[float], float]
ScalarSpaceTimeFunc = Callable[[float, np.ndarray], np.ndarray]


# -----------------------------------------------------------------------------
# Finite element forms.
# -----------------------------------------------------------------------------
@BilinearForm
def mass(u, v, w):
    r"""Standard L2 mass form: \int u v."""
    return u * v


@BilinearForm
def laplace(u, v, w):
    r"""Standard diffusion form: \int grad(u) . grad(v)."""
    return dot(grad(u), grad(v))


# -----------------------------------------------------------------------------
# Parameter container.
# -----------------------------------------------------------------------------
@dataclass
class SITParameters:
    """Collection of all model parameters as callables.

    Every parameter is stored as a function so that replacing constants by
    time-dependent coefficients later will require minimal code changes.

    For the carrying capacity k, the callable accepts

        t : float
            Current time.
        x : np.ndarray
            Array of nodal coordinates with shape (2, n_nodes), where x[0, :]
            are x-coordinates and x[1, :] are y-coordinates.

    and returns an array of shape (n_nodes,).
    """
    k: ScalarSpaceTimeFunc
    d_F: ScalarTimeFunc
    gamma: ScalarTimeFunc
    zeta: ScalarTimeFunc
    eta: float
    d_M: ScalarTimeFunc
    mu_M: ScalarTimeFunc
    r: float
    
    def briere(self, T, q, Tmin, Tmax):
        return q * T * (T - Tmin) * np.sqrt(Tmax - T)
    
    def quadratic(self, T, c, Tmin, Tmax):
        return c * (T - Tmin) * (T - Tmax)
    
    def temp(self, t):
        A = np.array(
            [0,
             1.9236681077,
             0.3403149062,
             0.2768740260,
             0.1200293376
            ])
        
        B = np.array(
            [20.3739251839,
             1.3960271141,
             -0.0590627008,
             0.0083445950,
             0.0284346876])
        
        sines = np.array([np.sin(2 * np.pi * k * t / 365) for k in range(len(A))])
        cosines = np.array([np.cos(2 * np.pi * k * t / 365) for k in range(len(B))])
        
        return np.dot(A, sines) + np.dot(B, cosines)
    
    def phi(self, t):
        T = self.temp(t)
        
        # There are two options given for this function. I used the first.
        q = 0.00856
        Tmin = 14.58
        Tmax = 34.61
        
        return self.briere(T, q, Tmin, Tmax)
    
    def sigma_E(self, t):
        T = self.temp(t)
        
        # There are two options given for this function. I used the first.
        q = 0.0000786
        Tmin = 11.36
        Tmax = 39.17
        
        return self.briere(T, q, Tmin, Tmax)
    
    def mu_E(self, t):
        T = self.temp(t)
        
        # There are two options given for this function. I used the first.
        c = -0.00599
        Tmin = 13.56
        Tmax = 38.29
        
        p_EA = self.quadratic(T, c, Tmin, Tmax)
        
        return -self.sigma_E(t) * np.log(p_EA)
    
    def mu_F(self, t):
        T = self.temp(t)
        
        # There are three options for this function. I used the first.
        
        c = -0.148
        Tmin = 9.16
        Tmax = 37.73
        
        lf = self.quadratic(T, c, Tmin, Tmax)
        
        return 1/lf


@dataclass
class SolverState:
    """Finite element solution vectors and metadata for the time march."""

    times: np.ndarray
    E_history: np.ndarray
    F_history: np.ndarray
    M_history: np.ndarray
    mesh: MeshTri
    node_coords: np.ndarray
    triangles: np.ndarray
    origin_node_index: int
    interval_end_indices: np.ndarray
    release_times: np.ndarray
    release_sizes: np.ndarray


# -----------------------------------------------------------------------------
# Default constant parameter functions.
# -----------------------------------------------------------------------------
def constant_time_function(value: float) -> ScalarTimeFunc:
    """Return a function of time that always yields a constant scalar value."""
    return lambda t: float(value)


def constant_space_time_function(value: float) -> ScalarSpaceTimeFunc:
    """Return a function of (time, nodes) that always yields a constant field."""

    def _func(t: float, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[1], float(value), dtype=float)

    return _func


def default_parameters() -> SITParameters:
    """Build the default constant parameter set requested by the user."""
    return SITParameters(
        k=constant_space_time_function(0.18),
        d_F=constant_time_function(1080.0),
        gamma=constant_time_function(0.00332),
        zeta=constant_time_function(2.0),
        r=0.5,
        eta=0.8,
        d_M=constant_time_function(1080.0),
        mu_M=constant_time_function(0.104),
    )


# -----------------------------------------------------------------------------
# Mesh / FEM helpers.
# -----------------------------------------------------------------------------
def meshtri_from_mesh_data(mesh_data: dict[str, Any]) -> MeshTri:
    """Construct a scikit-fem MeshTri from the dictionary returned by Gmsh.

    scikit-fem expects

        p : array of shape (2, n_nodes)
            node coordinates
        t : array of shape (3, n_elements)
            triangle connectivity

    while our Gmsh helper returns the more plotting-friendly shapes

        node_coords : (n_nodes, 2)
        triangles   : (n_elements, 3)
    """
    p = np.asarray(mesh_data["node_coords"], dtype=float).T
    t = np.asarray(mesh_data["triangles"], dtype=int).T
    return MeshTri(p, t)


def find_origin_node(node_coords: np.ndarray) -> int:
    """Return the index of the mesh node closest to the origin.

    The updated Bellavista mesher is intended to embed the point (0, 0) into the
    mesh.  In that ideal case, this function should return the exact release
    point.  If roundoff or meshing subtleties intervene, it still returns the
    closest available nodal degree of freedom.
    """
    distances = np.linalg.norm(node_coords, axis=1)
    return int(np.argmin(distances))


def assemble_reference_matrices(mesh: MeshTri) -> tuple[csr_matrix, csr_matrix, np.ndarray]:
    """Assemble the finite element mass and stiffness matrices.

    Returns
    -------
    Mmat : csr_matrix
        Consistent mass matrix.
    Kmat : csr_matrix
        Stiffness matrix corresponding to the Laplacian.
    lumped_mass : np.ndarray
        Lumped mass vector, obtained by summing each row of the consistent mass
        matrix.  This is convenient for nodewise reaction terms and the impulse
        jump in M.
    """
    basis = Basis(mesh, ElementTriP1())
    Mmat = asm(mass, basis).tocsr()
    Kmat = asm(laplace, basis).tocsr()
    lumped_mass = np.asarray(Mmat.sum(axis=1)).ravel()
    return Mmat, Kmat, lumped_mass


# -----------------------------------------------------------------------------
# Initial data.
# -----------------------------------------------------------------------------
def sit_free_equilibrium_at_nodes(
    params: SITParameters,
    node_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the SIT-free equilibrium requested by the user."""
    t0 = 0.0
    x = node_coords.T

    gamma0 = params.gamma(t0)
    muF0 = params.mu_F(t0)
    sigmaE0 = params.sigma_E(t0)
    muE0 = params.mu_E(t0)
    phi0 = params.phi(t0)
    r = params.r
    k0 = params.k(t0, x)

    k0 = np.asarray(k0, dtype=float)
    if k0.ndim > 0:
        if not np.allclose(k0, k0.flat[0]):
            raise ValueError(
                "The new equilibrium depends on k. Since you requested constant "
                "arrays, k(t0, x) must be spatially constant."
            )
        k0 = float(k0.flat[0])
    else:
        k0 = float(k0)

    a = (1.0 - r) * r * sigmaE0 * phi0
    b = k0 * (1.0 - r) * (muE0 * muF0 + r * sigmaE0 * (muF0 - phi0))
    c = gamma0 * k0 * muE0 * muF0 * sigmaE0

    roots = np.roots([a, b, c])
    if np.any(np.abs(np.imag(roots)) > 1e-10):
        raise ValueError("The larger root for E_star is complex.")

    roots = np.real(roots)
    E_star = float(np.max(roots))

    if E_star < 0:
        raise ValueError("The larger root for E_star is negative.")

    denom_F = muF0 * (gamma0 * sigmaE0 + (1.0 - r) * E_star)
    if np.isclose(denom_F, 0.0):
        raise ValueError(
            "The SIT-free equilibrium for F is singular because "
            "mu_F * (gamma * sigma_E + (1-r) * E_star) = 0."
        )

    F_star = E_star * (1.0 - r) * r * sigmaE0 / denom_F

    E0 = np.full(node_coords.shape[0], E_star, dtype=float)
    F0 = np.full(node_coords.shape[0], F_star, dtype=float)

    return E0, F0

# -----------------------------------------------------------------------------
# Time-stepping helpers.
# -----------------------------------------------------------------------------
def apply_m_impulse(Mvec: np.ndarray, origin_node: int, lumped_mass: np.ndarray, amount: float) -> np.ndarray:
    """Apply one sterile-male release as an instantaneous jump.

    In lumped-mass form, the jump condition is approximated by

        m_j (M_j^+ - M_j^-) = S,

    at the source node j, so the nodal value increases by S / m_j.

    Parameters
    ----------
    Mvec : np.ndarray
        Current nodal values of M just before the impulse.
    origin_node : int
        Index of the source node.
    lumped_mass : np.ndarray
        Lumped mass vector.
    amount : float
        Release size S_i.
    """
    updated = Mvec.copy()
    updated[origin_node] += float(amount) / lumped_mass[origin_node]
    return updated


def step_E(
    E: np.ndarray,
    F: np.ndarray,
    Msterile: np.ndarray,
    t: float,
    dt: float,
    params: SITParameters,
    node_coords: np.ndarray,
) -> np.ndarray:
    """Advance the egg equation one time step with a forward Euler update."""
    x = node_coords.T
    phi = params.phi(t)
    k = params.k(t, x)
    gamma = params.gamma(t)
    sigma_E = params.sigma_E(t)
    mu_E = params.mu_E(t)
    r = params.r
    eta = params.eta

    denom = gamma * sigma_E + ((1.0 - r) * E + eta * Msterile)
    denom = np.maximum(denom, 1e-12)

    rhs = (
        phi * F * (1.0 - E / k)
        - ((sigma_E * r * ((1.0 - r) * E + eta * Msterile) / denom) + mu_E) * E
    )

    E_new = E + dt * rhs
    return np.maximum(E_new, 0.0)


def step_F(
    F: np.ndarray,
    E: np.ndarray,
    Msterile: np.ndarray,
    t: float,
    dt: float,
    params: SITParameters,
    lumped_mass: np.ndarray,
    Kmat: csr_matrix,
) -> np.ndarray:
    gamma = params.gamma(t)
    sigma_E = params.sigma_E(t)
    r = params.r
    eta = params.eta
    mu_F = params.mu_F(t)
    d_F = params.d_F(t)

    denom = gamma * sigma_E + ((1.0 - r) * E + eta * Msterile)
    denom = np.maximum(denom, 1e-12)

    source = sigma_E * r * (1.0 - r) * E**2 / denom

    A = diags(lumped_mass / dt + lumped_mass * mu_F) + d_F * Kmat
    rhs = (lumped_mass / dt) * F + lumped_mass * source

    F_new = splu(A.tocsc()).solve(rhs)
    return np.maximum(F_new, 0.0)


def step_M(
    Msterile: np.ndarray,
    t: float,
    dt: float,
    params: SITParameters,
    lumped_mass: np.ndarray,
    Kmat: csr_matrix,
) -> np.ndarray:
    """Advance the sterile-male PDE one time step between impulses.

    Between impulses, M solves the linear reaction-diffusion equation

        dM/dt - d_M(t) Delta M = -mu_M(t) M.

    In lumped-mass backward Euler form,

        [diag(m/dt + m * mu_M) + d_M K] M^{n+1} = diag(m/dt) M^n.
    """
    d_M = params.d_M(t)
    mu_M = params.mu_M(t)

    A = diags(lumped_mass / dt + lumped_mass * mu_M) + d_M * Kmat
    rhs = (lumped_mass / dt) * Msterile

    M_new = splu(A.tocsc()).solve(rhs)
    return np.maximum(M_new, 0.0)


def release_schedule(tau: float = 7.0, K: int = 52, release_size: float = 100000.0) -> tuple[np.ndarray, np.ndarray]:
    """Create the default weekly release schedule requested by the user."""
    times = tau * np.arange(1, K + 1, dtype=float)
    sizes = np.full(K, float(release_size), dtype=float)
    return times, sizes


# -----------------------------------------------------------------------------
# Main solver.
# -----------------------------------------------------------------------------
def solve_sit_system(
    json_path: str | Path,
    params: SITParameters | None = None,
    h: float = 10.0,
    refinement_parameters: tuple[float, float] = (50.0, 1.0),
    tau: float = 7.0,
    K: int = 52,
    release_sizes: np.ndarray | None = None,
    dt: float = 0.1,
    store_every_step: bool = True,
) -> SolverState:
    """Solve the Bellavista SIT system on the requested time horizon.

    Parameters
    ----------
    json_path : str or Path
        Path to the Bellavista polygon JSON/GeoJSON file.
    params : SITParameters or None
        Model parameters.  If None, the default constant values are used.
    h : float, optional
        Baseline mesh size in meters.
    refinement_parameters : tuple(float, float), optional
        Passed through to ``build_polygon_mesh``.
    tau : float, optional
        Time between consecutive releases, in days.
    K : int, optional
        Number of releases.
    release_sizes : np.ndarray or None, optional
        Array of length K containing the release sizes S_i.  If None, all
        releases are set to 100000.
    dt : float, optional
        Internal time step for the PDE / ODE solve on each 7-day interval.
    store_every_step : bool, optional
        If True, store the full time history.  If False, only interval-end data
        are kept.  For the current problem sizes, storing every step is usually
        fine and is more convenient for plotting.
    """
    if params is None:
        params = default_parameters()

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if K < 1:
        raise ValueError("K must be at least 1.")

    if release_sizes is None:
        _, release_sizes = release_schedule(tau=tau, K=K, release_size=100000.0)
    else:
        release_sizes = np.asarray(release_sizes, dtype=float)
        if len(release_sizes) != K:
            raise ValueError("release_sizes must have length K.")

    release_times = tau * np.arange(1, K + 1, dtype=float)
    final_time = float(K * tau)

    # ------------------------------------------------------------------
    # Build the Bellavista mesh.
    # ------------------------------------------------------------------
    vertices = latlon_to_cartesian(json_path)
    mesh_data = build_polygon_mesh(vertices, h=h, refinement_parameters=refinement_parameters)
    mesh = meshtri_from_mesh_data(mesh_data)

    node_coords = np.asarray(mesh_data["node_coords"], dtype=float)
    triangles = np.asarray(mesh_data["triangles"], dtype=int)
    origin_node = find_origin_node(node_coords)

    # ------------------------------------------------------------------
    # Assemble FEM matrices once, because the geometry is static.
    # ------------------------------------------------------------------
    _, Kmat, lumped_mass = assemble_reference_matrices(mesh)

    # ------------------------------------------------------------------
    # Initial conditions.
    # ------------------------------------------------------------------
    E, F = sit_free_equilibrium_at_nodes(params, node_coords)
    Msterile = np.zeros_like(E)

    # ------------------------------------------------------------------
    # Time storage.
    # ------------------------------------------------------------------
    times = [0.0]
    E_hist = [E.copy()]
    F_hist = [F.copy()]
    M_hist = [Msterile.copy()]
    interval_end_indices: list[int] = []

    current_time = 0.0
    release_counter = 0

    # ------------------------------------------------------------------
    # March interval by interval.  On each interval we solve only up to the
    # next release time, then apply the impulsive jump in M.
    # ------------------------------------------------------------------
    while current_time < final_time - 1e-12:
        next_release_time = release_times[release_counter]

        # March from current_time up to next_release_time using steps of size dt,
        # with one final shorter step if necessary.
        while current_time < next_release_time - 1e-12:
            this_dt = min(dt, next_release_time - current_time)

            E_new = step_E(E, F, Msterile, current_time, this_dt, params, node_coords)
            F_new = step_F(F, E, Msterile, current_time, this_dt, params, lumped_mass, Kmat)
            M_new = step_M(Msterile, current_time, this_dt, params, lumped_mass, Kmat)

            current_time += this_dt
            E, F, Msterile = E_new, F_new, M_new

            if store_every_step:
                times.append(current_time)
                E_hist.append(E.copy())
                F_hist.append(F.copy())
                M_hist.append(Msterile.copy())

        # Record the state immediately before the impulse if we are only storing
        # interval endpoints.
        if not store_every_step:
            times.append(current_time)
            E_hist.append(E.copy())
            F_hist.append(F.copy())
            M_hist.append(Msterile.copy())

        # Apply the impulse exactly at t = (release_counter + 1) * tau.
        Msterile = apply_m_impulse(Msterile, origin_node, lumped_mass, release_sizes[release_counter])

        # Store the post-impulse state, because this is the initial data for the
        # next 7-day interval.
        times.append(current_time)
        E_hist.append(E.copy())
        F_hist.append(F.copy())
        M_hist.append(Msterile.copy())
        interval_end_indices.append(len(times) - 1)

        release_counter += 1

    return SolverState(
        times=np.asarray(times, dtype=float),
        E_history=np.asarray(E_hist, dtype=float),
        F_history=np.asarray(F_hist, dtype=float),
        M_history=np.asarray(M_hist, dtype=float),
        mesh=mesh,
        node_coords=node_coords,
        triangles=triangles,
        origin_node_index=origin_node,
        interval_end_indices=np.asarray(interval_end_indices, dtype=int),
        release_times=release_times,
        release_sizes=np.asarray(release_sizes, dtype=float),
    )


# -----------------------------------------------------------------------------
# Plotting helpers.
# -----------------------------------------------------------------------------
def plot_interval_end_means(state: SolverState, show: bool = True) -> tuple[plt.Figure, np.ndarray]:
    """Plot the spatial means of E, F, and M after each weekly release.

    This is often a good first diagnostic: it makes it easy to see whether the
    solution is blowing up, decaying, or settling into a repeating pattern.
    """
    idx = state.interval_end_indices
    times = state.times[idx]

    E_means = state.E_history[idx].mean(axis=1)
    F_means = state.F_history[idx].mean(axis=1)
    M_means = state.M_history[idx].mean(axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(times, E_means, marker="o")
    axes[0].set_ylabel("mean E")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, F_means, marker="o")
    axes[1].set_ylabel("mean F")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, M_means, marker="o")
    axes[2].set_ylabel("mean M")
    axes[2].set_xlabel("time (days)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Spatial means after each weekly release")
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes

def plot_solution_with_time_slider(state, field="M", shading="gouraud"):
    global _time_slider, _field_radio

    field = field.upper()
    if field not in ("E", "F", "M"):
        raise ValueError("field must be one of 'E', 'F', or 'M'.")

    x = state.node_coords[:, 0]
    y = state.node_coords[:, 1]
    tri = state.triangles
    times = state.times

    histories = {
        "E": state.E_history,
        "F": state.F_history,
        "M": state.M_history,
    }

    # Precompute a fixed color scale for each field over all times.
    field_max = {}
    for name, hist in histories.items():
        vmax = max(np.max(values) for values in hist)
        if vmax <= 0:
            vmax = 1e-12
        field_max[name] = vmax
    
    print(field_max)
    
    current_field = {"name": field}
    z0 = histories[current_field["name"]][0]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.08, 0.20, 0.72, 0.72])
    ax_slider = fig.add_axes([0.08, 0.08, 0.72, 0.05])
    ax_radio = fig.add_axes([0.84, 0.55, 0.12, 0.20])

    # Set fixed color limits for the initial field.
    tpc = ax.tripcolor(
        x,
        y,
        tri,
        z0,
        shading=shading,
        vmin=0.0,
        vmax=field_max[current_field["name"]],
    )
    cbar = fig.colorbar(tpc, ax=ax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = ax.set_title(f"{current_field['name']} at t = {times[0]:.2f} days")

    if hasattr(state, "origin_node_index"):
        j = state.origin_node_index
        ax.plot(x[j], y[j], "ro", markersize=5, label="release point")
        ax.legend(loc="upper right")

    _time_slider = Slider(
        ax=ax_slider,
        label="Time index",
        valmin=0,
        valmax=len(times) - 1,
        valinit=0,
        valstep=1,
    )

    _field_radio = RadioButtons(
        ax_radio,
        ("E", "F", "M"),
        active=("E", "F", "M").index(field),
    )

    def update_plot(idx, update_color_scale=False):
        idx = int(idx)
        values = histories[current_field["name"]][idx]

        # Update the plotted values only.
        tpc.set_array(values)

        # Only change the color scale when switching fields.
        if update_color_scale:
            tpc.set_clim(vmin=0.0, vmax=field_max[current_field["name"]])
            cbar.update_normal(tpc)

        title.set_text(f"{current_field['name']} at t = {times[idx]:.2f} days")
        fig.canvas.draw_idle()

    def on_slider_change(val):
        update_plot(val, update_color_scale=False)

    def on_radio_click(label):
        current_field["name"] = label
        update_plot(_time_slider.val, update_color_scale=True)

    _time_slider.on_changed(on_slider_change)
    _field_radio.on_clicked(on_radio_click)

    plt.show()


def plot_final_fields(state: SolverState, show: bool = True) -> tuple[plt.Figure, np.ndarray]:
    """Plot the final spatial distributions of E, F, and M on the mesh."""
    E = state.E_history[-1]
    F = state.F_history[-1]
    Msterile = state.M_history[-1]
    x = state.node_coords[:, 0]
    y = state.node_coords[:, 1]
    tri = state.triangles

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for ax, field, title in zip(axes, [E, F, Msterile], ["Final E", "Final F", "Final M"]):
        tpc = ax.tripcolor(x, y, tri, field, shading="gouraud")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(tpc, ax=ax)

    if show:
        plt.show()

    return fig, axes

def plot_total_populations(state: SolverState, show: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """Plot total egg and female populations versus time.

    Totals are approximated using the lumped finite-element mass:
        total(u) ≈ sum_j m_j u_j
    """
    _, _, lumped_mass = assemble_reference_matrices(state.mesh)

    total_E = state.E_history @ lumped_mass
    total_F = state.F_history @ lumped_mass

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(state.times, total_E, label="Total E")
    ax.plot(state.times, total_F, label="Total F")

    ax.set_xlabel("time (days)")
    ax.set_ylabel("total population")
    ax.set_title("Total egg and female populations over time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
    

# -----------------------------------------------------------------------------
# Command-line interface.
# -----------------------------------------------------------------------------
def main() -> None:
    """Run the Bellavista SIT solver from the command line."""
    parser = argparse.ArgumentParser(
        description="Solve the Bellavista SIT PDE/ODE system using scikit-fem."
    )
    parser.add_argument(
        "json_file",
        type=str,
        nargs="?",
        default="bellavista.json",
        help="Path to the Bellavista JSON/GeoJSON boundary file.",
    )
    parser.add_argument("--h", type=float, default=10.0, help="Background mesh size in meters.")
    parser.add_argument("--refinement-radius", type=float, default=50.0, help="Radius of local refinement around (0,0).")
    parser.add_argument("--refinement-degree", type=float, default=1.0, help="Max refinement degree near (0,0).")
    parser.add_argument("--dt", type=float, default=0.05, help="Internal time step in days.")
    parser.add_argument("--tau", type=float, default=7.0, help="Release period in days.")
    parser.add_argument("--K", type=int, default=52, help="Number of weekly releases.")
    parser.add_argument("--show-mesh", action="store_true", help="Plot the Bellavista mesh before solving.")
    parser.add_argument("--show-results", action="store_true", help="Plot summary diagnostics and final fields.")
    args = parser.parse_args()

    # Easy-to-edit release vector requested by the user.
    S_i = np.full(args.K, 100000.0, dtype=float)

    params = default_parameters()

    state = solve_sit_system(
        json_path=args.json_file,
        params=params,
        h=args.h,
        refinement_parameters=(args.refinement_radius, args.refinement_degree),
        tau=args.tau,
        K=args.K,
        release_sizes=S_i,
        dt=args.dt,
        store_every_step=True,
    )

    # Print a small textual summary.
    print(f"Mesh nodes                 : {state.node_coords.shape[0]}")
    print(f"Triangles                  : {state.triangles.shape[0]}")
    print(f"Origin node index          : {state.origin_node_index}")
    print(f"Origin node coordinates    : {state.node_coords[state.origin_node_index]}")
    print(f"Final time (days)          : {state.times[-1]:.6f}")
    print(f"Final mean E               : {state.E_history[-1].mean():.6f}")
    print(f"Final mean F               : {state.F_history[-1].mean():.6f}")
    print(f"Final mean M               : {state.M_history[-1].mean():.6f}")
    print(f"Final max M                : {state.M_history[-1].max():.6f}")

    if args.show_mesh:
        vertices = latlon_to_cartesian(args.json_file)
        mesh_data = build_polygon_mesh(
            vertices,
            h=args.h,
            refinement_parameters=(args.refinement_radius, args.refinement_degree),
        )
        plot_mesh(mesh_data, show_nodes=False, show=True)

    plot_solution_with_time_slider(state)
    plot_total_populations(state)
    return state


if __name__ == "__main__":
    mesh_data = main()
