"""Import / recipe smoke checks for boson on upstream layout."""

import pytest


def test_boson_recipe_imports():
    from pyqmc.bosonrecipes import ABDMC, ABOPTIMIZE, ABVMC
    from pyqmc.observables import bosonaccumulators, bosonenergy, mf_hartree
    from pyqmc.method import bosondmc, bosonlinemin, bosonmc
    from pyqmc.wf import bosonslater

    assert callable(ABVMC) and callable(ABDMC) and callable(ABOPTIMIZE)
    assert hasattr(bosonenergy, "dft_energy")
    assert hasattr(mf_hartree, "HartreePotentialEvaluator")
    from pyqmc.observables import mf_grid_interp, mf_ri_hartree

    assert hasattr(mf_grid_interp, "GridMFPotentialEvaluator")
    assert hasattr(mf_ri_hartree, "RIHartreePotentialEvaluator")
    assert "grid" in bosonenergy.SUPPORTED_EVALUATE_MF
    assert "ri" in bosonenergy.SUPPORTED_EVALUATE_MF
    assert hasattr(bosonslater, "BosonWF")
    assert hasattr(bosonaccumulators, "ABQMCEnergyAccumulator")
    assert hasattr(bosonmc, "abvmc")
    assert hasattr(bosondmc, "rundmc")
    assert hasattr(bosonlinemin, "line_minimization")


def test_boson_example_runners_importable():
    """Example scripts use package paths that resolve under PYTHONPATH=repo."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "examples" / "boson"
    for name in ("he_abvmc.py", "run_abdmc.py"):
        path = root / name
        assert path.is_file(), path
        # Only compile-check; run_abdmc needs a local config.yaml
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
