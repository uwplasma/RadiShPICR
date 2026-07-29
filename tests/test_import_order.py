import os
import subprocess
import sys


def test_package_defaults_to_jax_x64_when_environment_is_unset():
    env = os.environ.copy()
    env.pop("JAX_ENABLE_X64", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import RadiShPICR, jax; "
                "print(jax.config.read('jax_enable_x64'))"
            ),
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "True"


def test_package_respects_explicit_jax_x64_environment_setting():
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "0"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import RadiShPICR, jax; "
                "print(jax.config.read('jax_enable_x64'))"
            ),
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_radial_electric_solver_imports_without_relativity_metric_cycle():
    from RadiShPICR.ConstraintBasedRelativity import (
        calculate_metric,
        step,
        step_rk4,
        step_rk4_with_metric,
    )

    assert callable(step)
    assert callable(step_rk4)
    assert callable(step_rk4_with_metric)
    assert callable(calculate_metric)


def test_formulation_local_evolution_imports_are_available():
    from RadiShPICR.ConstraintBasedRelativity.evolve import (
        step,
        step_rk4,
        step_rk4_with_metric,
    )
    from RadiShPICR.Z4C import (
        advance_vacuum_steps,
        particles_rk4_step,
        rk4_step,
    )

    assert callable(step)
    assert callable(step_rk4)
    assert callable(step_rk4_with_metric)
    assert callable(advance_vacuum_steps)
    assert callable(rk4_step)
    assert callable(particles_rk4_step)


def test_top_level_evolve_keeps_constraint_based_compatibility_imports():
    from RadiShPICR.ConstraintBasedRelativity.evolve import step, step_rk4
    from RadiShPICR.evolve import step as compat_step
    from RadiShPICR.evolve import step_rk4 as compat_step_rk4

    assert compat_step is step
    assert compat_step_rk4 is step_rk4
