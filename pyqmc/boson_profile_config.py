"""
Lightweight switches for boson DMC / ABCDMC wall-clock profiling (no heavy imports).

Call :func:`configure` after loading ``config.yaml`` so Slurm/srun does not need to
forward custom environment variables. Arguments that are ``None`` leave that
setting unchanged (still env-based). Pass explicit ``False`` / ``0`` to disable.

Example::

    import yaml
    from pyqmc import boson_profile_config

    with open(\"config.yaml\") as f:
        config = yaml.safe_load(f)
    boson_profile_config.configure(
        enabled=config.get(\"profile_boson_dmc\"),
        print_every=config.get(\"profile_abcdmc_print_every\"),
    )

If ``configure`` is never called, behavior falls back to
``PYQMC_PROFILE_BOSON_DMC``, ``PYQMC_PROFILE_ABCDMC``, and
``PYQMC_PROFILE_ABCDMC_PRINT_EVERY`` (truthy: 1/true/yes/on).
"""
import os
from typing import Optional

_cfg_enabled: Optional[bool] = None
_cfg_print_every: Optional[int] = None


def configure(enabled=None, print_every=None):
    """Set overrides from YAML (or elsewhere). ``None`` = leave as-is / use env."""
    global _cfg_enabled, _cfg_print_every
    if enabled is not None:
        _cfg_enabled = bool(enabled)
    if print_every is not None:
        try:
            _cfg_print_every = max(0, int(print_every))
        except (TypeError, ValueError):
            _cfg_print_every = 0


def reset():
    """Clear YAML overrides; use only environment variables again."""
    global _cfg_enabled, _cfg_print_every
    _cfg_enabled = None
    _cfg_print_every = None


def _env_truthy(key):
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    if _cfg_enabled is not None:
        return bool(_cfg_enabled)
    return _env_truthy("PYQMC_PROFILE_BOSON_DMC") or _env_truthy("PYQMC_PROFILE_ABCDMC")


def print_interval() -> int:
    if _cfg_print_every is not None:
        return int(_cfg_print_every)
    try:
        return max(0, int(os.environ.get("PYQMC_PROFILE_ABCDMC_PRINT_EVERY", "0")))
    except ValueError:
        return 0
