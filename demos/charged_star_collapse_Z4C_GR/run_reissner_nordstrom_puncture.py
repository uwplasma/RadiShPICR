"""Evolve the paper's single RN puncture with the star's initial mass and charge.

Initial data: arXiv:1311.6483v2, equations (2.8)-(2.13), single puncture u=1.
Units are G=c=1; Q is physical charge and q=Q/sqrt(4*pi*epsilon_0).
The conserved quantity is electric flux, not the coordinate component E_r.
"""

import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from RadiShPICR.Z4C.curvature_invariants import misner_sharp_mass
from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.electric_field import compute_electrovac_matter_terms
from RadiShPICR.Z4C.energy_momentum_tensor import compute_hamiltonian_constraint
from RadiShPICR.Z4C.time_evolve import electrovac_rk4_step
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


# Edit these settings before running the script.
INPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/z4c_charged_star_schwarzschild"
OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/reissner_nordstrom_puncture_initial_mass_charge"
DURATION = 150.0  # Final evolution t/M.
CFL = 0.2
SNAPSHOT_COUNT = 500
PUNCTURE_CHI_FLOOR = 1e-12  # First three cells at RK stages; 0 disables it.

# None uses the value recorded in the charged-star metadata.
ZERO_SHIFT = None  # Set 1 to freeze the initial shift or 0 to evolve it.
KAPPA = None
ETA = None
NU = None

SNAPSHOT_NAMES = {
    "alpha": "alpha", "beta": "beta", "chi": "chi",
    "conformal_grr": "conformal_grr", "conformal_gt": "conformal_gT",
    "Kh": "Kh", "Arr": "Arr", "At": "AT", "theta": "theta", "Gamma": "Gamma",
}


def read_initial_data(input_directory):
    """Read the initial gravitational mass, physical charge, grid, and gauge."""
    with np.load(Path(input_directory) / "initial_data_metadata.npz") as data:
        parameters = {name: float(data[name]) for name in (
            "total_mass", "initial_charge", "epsilon_0", "kappa", "eta", "nu",
        )}
        parameters["zero_shift"] = int(data["zero_shift"])
        r = np.asarray(data["z4c_r"], dtype=float)

    mass, charge = parameters["total_mass"], parameters["initial_charge"]
    epsilon_0 = parameters["epsilon_0"]
    if (not all(np.isfinite(value) for value in parameters.values())
            or epsilon_0 <= 0.0 or mass <= 0.0
            or abs(charge) / np.sqrt(4.0 * np.pi * epsilon_0) >= mass):
        raise ValueError("Require finite subextremal RN data: M > |Q|/sqrt(4*pi*epsilon_0).")
    # Origin parity and the existing eight-point boundary stencils need this grid.
    if r.ndim != 1 or r.size < 8:
        raise ValueError("The puncture requires at least eight radial cells.")
    dr = r[1] - r[0]
    if (not np.all(np.isfinite(r)) or dr <= 0.0
            or not np.allclose(np.diff(r), dr) or not np.isclose(r[0], 0.5 * dr)):
        raise ValueError("z4c_r must be a uniform half-cell-centered radial grid.")
    return r, parameters


def initialize_puncture(r, mass, charge, epsilon_0=1.0, kappa=1.0, eta=2.0, nu=0.02):
    """Exact time-symmetric RN data, with the OS precollapsed lapse choice."""
    r = jnp.asarray(r)
    zeros, ones = jnp.zeros_like(r), jnp.ones_like(r)
    geometric_charge_squared = charge**2 / (4.0 * jnp.pi * epsilon_0)
    psi_squared = (1.0 + mass / (2.0 * r))**2 - geometric_charge_squared / (4.0 * r**2)

    return Z4C_Metric(
        alpha=psi_squared**-1, beta=zeros, chi=psi_squared**-2,
        conformal_grr=ones, conformal_gt=ones,
        Kh=zeros, Arr=zeros, At=zeros, theta=zeros, Gamma=zeros,
        r=r, dr=r[1] - r[0], kappa=jnp.asarray(kappa),
        eta=jnp.asarray(eta), nu=jnp.asarray(nu),
    )


def puncture_diagnostics(metric, charge, epsilon_0):
    chi, grr, gt, r = metric.chi, metric.conformal_grr, metric.conformal_gt, metric.r
    R = r * jnp.sqrt(gt / chi)
    E_r = charge * jnp.sqrt(chi * grr) / (4.0 * jnp.pi * epsilon_0 * r**2 * gt)
    E_normal = E_r * jnp.sqrt(chi / grr)
    radial_flux = r**2 * gt * E_r / jnp.sqrt(chi * grr)
    charge_flux = 4.0 * jnp.pi * epsilon_0 * radial_flux
    volume_factor = r**2 * jnp.sqrt(grr) * gt / chi**1.5

    # Constant inner flux continues through the puncture. Regular-origin odd
    # flux parity would manufacture a charge-density spike in the first cells.
    gauss = epsilon_0 * first_derivative(radial_flux, metric.dr, parity=1) / volume_factor
    matter = compute_electrovac_matter_terms(metric, charge, epsilon_0)
    physical_K = metric.Kh + 2.0 * metric.theta
    hamiltonian = compute_hamiltonian_constraint(metric._replace(Kh=physical_K)) - 16.0 * jnp.pi * matter.rho
    mass_ms = misner_sharp_mass(metric)
    return {
        "areal_radius": R, "E_r": E_r, "E_normal": E_normal,
        "charge_flux": charge_flux, "em_energy_density": matter.rho,
        "charge_density": jnp.zeros_like(r), "mass_density": jnp.zeros_like(r),
        "Srr": matter.Srr, "ST": matter.Stt, "Sr": matter.Sr,
        "K": physical_K, "misner_sharp_mass": mass_ms,
        "rn_mass_from_flux": mass_ms + charge_flux**2 / (8.0 * jnp.pi * epsilon_0 * R),
        "gauss_residual": gauss, "hamiltonian_constraint": hamiltonian,
    }


def save_snapshot(metric, directory, step, evolution_time, mass, charge, epsilon_0):
    fields = {key: np.asarray(getattr(metric, name)) for name, key in SNAPSHOT_NAMES.items()}
    diagnostics = {key: np.asarray(value) for key, value in puncture_diagnostics(metric, charge, epsilon_0).items()}
    # Inner puncture and outer boundary derivative closures are excluded from
    # reported bulk norms, but their unmodified values remain in the snapshot.
    bulk = np.ones(metric.r.shape, dtype=bool)
    bulk[:3] = bulk[-3:] = False
    norms = {}
    for name in ("gauss_residual", "hamiltonian_constraint"):
        norms[name + "_bulk_linf"] = float(np.max(np.abs(diagnostics[name][bulk])))
        norms[name + "_bulk_rms"] = float(np.sqrt(np.mean(diagnostics[name][bulk]**2)))
    norms["charge_flux_error_linf"] = float(np.max(np.abs(diagnostics["charge_flux"] - charge)))
    norms["minimum_lapse"] = float(np.min(fields["alpha"]))
    norms["minimum_chi"] = float(np.min(fields["chi"]))
    np.savez_compressed(
        directory / f"metric_step_{step:06d}.npz", **fields, **diagnostics, **norms,
        r=np.asarray(metric.r), diagnostic_bulk_mask=bulk, step=step,
        time=evolution_time, evolution_time=evolution_time, time_over_mass=evolution_time / mass,
        total_mass=mass, initial_charge=charge, epsilon_0=epsilon_0,
        saved_coordinates="evolution", electric_field_update="constant_puncture_flux",
    )
    return norms


def advance_puncture_steps(metric, charge, dt, num_steps, epsilon_0=1.0,
                          zero_shift=0, puncture_chi_floor=0.0):
    """Return the last acceptable state, accepted step count, and failure code."""
    def advance_one_step(carry, unused):
        def advance(carry):
            metric, accepted, failure = carry
            trial = electrovac_rk4_step(
                metric, charge, dt, epsilon_0=epsilon_0, zero_shift=zero_shift,
                puncture_chi_floor=puncture_chi_floor,
            )
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(field)) for field in trial]))
            positive = jnp.all((trial.chi > 0.0) & (trial.conformal_grr > 0.0) & (trial.conformal_gt > 0.0))
            valid = finite & positive
            last_valid = jax.lax.cond(valid, lambda _: trial, lambda _: metric, operand=None)
            failure = jnp.where(valid, 0, jnp.where(finite, 2, 1)).astype(jnp.int32)
            return last_valid, accepted + valid.astype(jnp.int32), failure

        carry = jax.lax.cond(carry[2] == 0, advance, lambda state: state, carry)
        return carry, None

    state = (metric, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    state, _ = jax.lax.scan(advance_one_step, state, xs=None, length=num_steps)
    return state


def evolve_puncture(metric, mass, charge, epsilon_0, output_directory,
                    duration=150.0, cfl=0.2, snapshot_count=500, zero_shift=0,
                    provenance=None, puncture_chi_floor=1e-12):
    output_directory = Path(output_directory)
    # Never reuse a campaign directory, even if it only contains its metadata.
    output_directory.mkdir(parents=True, exist_ok=False)
    metric_directory = output_directory / "metric"
    metric_directory.mkdir()
    target_time = duration * mass
    num_steps = max(1, math.ceil(target_time / (cfl * float(metric.dr))))
    dt = target_time / num_steps
    interval = max(1, math.ceil(num_steps / snapshot_count))
    completed_steps, failure = 0, 0
    started = time.monotonic()
    summary = dict(provenance or {})
    summary.update(
        mass=mass, initial_charge=charge, epsilon_0=epsilon_0,
        geometric_charge=charge / np.sqrt(4.0 * np.pi * epsilon_0),
        mass_source="initial_gravitational_mass", charge_source="initial_charge",
        completed=False, target_time=target_time, duration_over_mass=duration,
        time_coordinate="evolution", dt=dt, cfl=cfl, snapshot_interval_steps=interval,
        num_cells=int(metric.r.size), dr=float(metric.dr), zero_shift=int(zero_shift),
        kappa=float(metric.kappa), eta=float(metric.eta), nu=float(metric.nu),
        metric_boundary="sommerfeld", initial_lapse="psi^-2",
        electric_field_update="constant_puncture_flux", initial_solver="analytic_single_puncture_u_1",
        initial_data_reference="arXiv:1311.6483v2 equations (2.8)-(2.13)",
        gauss_diagnostic="nonzero_inner_flux_even_continuation",
        diagnostic_bulk_excluded_cells_each_edge=3,
        puncture_chi_floor=puncture_chi_floor,
        puncture_chi_floor_cells=3 if puncture_chi_floor > 0.0 else 0,
        puncture_chi_floor_placement="each_RK_stage_and_final_state",
    )
    np.savez_compressed(
        output_directory / "initial_data_metadata.npz",
        **summary, total_mass=mass, z4c_r=np.asarray(metric.r),
    )
    advance = jax.jit(advance_puncture_steps, static_argnames=("num_steps",))
    with tqdm(total=num_steps, desc="evolving RN puncture") as progress:
        while True:
            norms = save_snapshot(metric, metric_directory, completed_steps, completed_steps * dt, mass, charge, epsilon_0)
            summary.update(
                completed=completed_steps == num_steps, final_step=completed_steps,
                final_time=completed_steps * dt, final_time_over_mass=completed_steps * dt / mass,
                final_evolution_time=completed_steps * dt, **norms,
                maximum_saved_charge_flux_error=max(summary.get("maximum_saved_charge_flux_error", 0.0), norms["charge_flux_error_linf"]),
                stop_reason=("nonfinite_state" if failure == 1 else "nonpositive_geometry") if failure else (
                    "target_time" if completed_steps == num_steps else "running"),
                first_invalid_step=completed_steps + 1 if failure else None,
                first_invalid_time=(completed_steps + 1) * dt if failure else None,
                wall_runtime_seconds=time.monotonic() - started,
            )
            (output_directory / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            if failure or completed_steps == num_steps:
                break
            chunk_steps = min(interval, num_steps - completed_steps)
            metric, accepted, failure = advance(metric, charge, dt, num_steps=chunk_steps,
                                               epsilon_0=epsilon_0, zero_shift=zero_shift,
                                               puncture_chi_floor=puncture_chi_floor)
            accepted, failure = int(accepted), int(failure)
            completed_steps += accepted
            progress.update(accepted)
    return summary


def main():
    if (not np.isfinite(DURATION) or not np.isfinite(CFL)
            or DURATION <= 0.0 or CFL <= 0.0 or SNAPSHOT_COUNT < 1):
        raise ValueError("DURATION, CFL, and SNAPSHOT_COUNT must be positive and finite.")
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {OUTPUT_DIRECTORY}")
    if not np.isfinite(PUNCTURE_CHI_FLOOR) or PUNCTURE_CHI_FLOOR < 0.0:
        raise ValueError("PUNCTURE_CHI_FLOOR must be finite and nonnegative.")

    jax.config.update("jax_enable_x64", True)
    r, parameters = read_initial_data(INPUT_DIRECTORY)
    for name, value in (("zero_shift", ZERO_SHIFT), ("kappa", KAPPA), ("eta", ETA), ("nu", NU)):
        if value is not None:
            parameters[name] = value
    mass, charge, epsilon_0 = (parameters[name] for name in ("total_mass", "initial_charge", "epsilon_0"))
    metric = initialize_puncture(r, mass, charge, epsilon_0,
                                parameters["kappa"], parameters["eta"], parameters["nu"])
    source = (INPUT_DIRECTORY / "initial_data_metadata.npz").resolve()
    print(f"RN reference: M={mass:.12g}, Q={charge:.12g}; source: {source}", flush=True)
    summary = evolve_puncture(
        metric, mass, charge, epsilon_0, OUTPUT_DIRECTORY,
        DURATION, CFL, SNAPSHOT_COUNT, parameters["zero_shift"],
        provenance={"source_initial_metadata": str(source), "mass_metadata_key": "total_mass"},
        puncture_chi_floor=PUNCTURE_CHI_FLOOR,
    )
    print(json.dumps(summary, indent=2))
    if not summary["completed"]:
        raise SystemExit("RN evolution stopped; the last valid checkpoint is preserved.")


if __name__ == "__main__":
    main()
