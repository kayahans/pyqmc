"""Independent LDA and PBE XC potentials (libxc vrho, not the full GGA KS potential).

LDA: Dirac/Slater x + VWN5 c.
PBE: PRL 77, 3865 (1996). vrho = ∂(nε)/∂n_σ at fixed ∇n (what ABCDMC/libxc return).
PBE correlation is built on PW92, not VWN.
"""

from __future__ import annotations

import numpy as np
from pyscf.dft import numint

# --- LDA-X (Dirac) ----------------------------------------------------------
def lda_vx_spin(rho_s):
    rho_s = np.maximum(np.asarray(rho_s, dtype=float), 1e-30)
    return -((6.0 / np.pi) ** (1.0 / 3.0)) * rho_s ** (1.0 / 3.0)


# --- VWN5 correlation -------------------------------------------------------
_VWN = dict(
    p=dict(A=0.0310907, x0=-0.10498, b=3.72744, c=12.9352),
    f=dict(A=0.01554535, x0=-0.32500, b=7.06042, c=18.0578),
    a=dict(A=-1.0 / (6.0 * np.pi**2), x0=-0.00475840, b=1.13107, c=13.0045),
)
_FPP0 = 1.709921


def _vwn_eps_depsdrs(rs, A, x0, b, c):
    Q = np.sqrt(4.0 * c - b * b)
    x = np.sqrt(rs)
    X = x * x + b * x + c
    X0 = x0 * x0 + b * x0 + c
    atan = np.arctan(Q / (2 * x + b))
    eps = A * (
        np.log(x * x / X)
        + 2 * b / Q * atan
        - (b * x0 / X0) * (np.log((x - x0) ** 2 / X) + 2 * (b + 2 * x0) / Q * atan)
    )
    d_atan = -2 * Q / ((2 * x + b) ** 2 + Q**2)
    dlog1 = 2 / x - (2 * x + b) / X
    dlog2 = 2 / (x - x0) - (2 * x + b) / X
    deps_dx = A * (
        dlog1 + 2 * b / Q * d_atan - (b * x0 / X0) * (dlog2 + 2 * (b + 2 * x0) / Q * d_atan)
    )
    return eps, deps_dx / (2 * x)


def _fzeta(z):
    c = 2 * (2 ** (1.0 / 3.0) - 1)
    return ((1 + z) ** (4.0 / 3.0) + (1 - z) ** (4.0 / 3.0) - 2) / c


def _dfzeta(z):
    c = 2 * (2 ** (1.0 / 3.0) - 1)
    return (4.0 / 3.0) * ((1 + z) ** (1.0 / 3.0) - (1 - z) ** (1.0 / 3.0)) / c


def _spin_interp_vc(rs, z, eps_fn):
    """v_c^σ from ε(r_s,ζ) using v = ε − (r_s/3)∂ε/∂r_s + (∂ε/∂ζ)(s−ζ)."""
    ep, dep = eps_fn(rs, "p")
    ef, def_ = eps_fn(rs, "f")
    ea, dea = eps_fn(rs, "a")
    fz, dfz = _fzeta(z), _dfzeta(z)
    z4 = z**4
    eps = ep + ea * (fz / _FPP0) * (1 - z4) + (ef - ep) * fz * z4
    deps_drs = dep + dea * (fz / _FPP0) * (1 - z4) + (def_ - dep) * fz * z4
    deps_dz = (
        ea * (dfz / _FPP0) * (1 - z4)
        + ea * (fz / _FPP0) * (-4 * z**3)
        + (ef - ep) * (dfz * z4 + fz * 4 * z**3)
    )
    base = eps - (rs / 3.0) * deps_drs
    return eps, base + deps_dz * (1 - z), base + deps_dz * (-1 - z)


def lda_vc_vwn(rho_a, rho_b):
    rho_a = np.maximum(np.asarray(rho_a, dtype=float), 1e-30)
    rho_b = np.maximum(np.asarray(rho_b, dtype=float), 1e-30)
    n = rho_a + rho_b
    z = np.clip((rho_a - rho_b) / n, -1 + 1e-12, 1 - 1e-12)
    rs = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)

    def eps_fn(rs, ch):
        return _vwn_eps_depsdrs(rs, **_VWN[ch])

    _, va, vb = _spin_interp_vc(rs, z, eps_fn)
    return va, vb


# --- PW92 LDA correlation (PBE's local piece) --------------------------------
_PW92 = dict(
    p=dict(A=0.031091, a1=0.21370, b1=7.5957, b2=3.5876, b3=1.6382, b4=0.49294),
    f=dict(A=0.015545, a1=0.20548, b1=14.1189, b2=6.1977, b3=3.3662, b4=0.62517),
    a=dict(A=0.016887, a1=0.11125, b1=10.357, b2=3.6231, b3=0.88026, b4=0.49671),
)


def _pw92_eps_depsdrs(rs, A, a1, b1, b2, b3, b4):
    rsqrt = np.sqrt(rs)
    q0 = -2.0 * A * (1.0 + a1 * rs)
    q1 = 2.0 * A * (b1 * rsqrt + b2 * rs + b3 * rs * rsqrt + b4 * rs * rs)
    ln = np.log(1.0 + 1.0 / q1)
    eps = q0 * ln
    dq0 = -2.0 * A * a1
    dq1 = 2.0 * A * (0.5 * b1 / rsqrt + b2 + 1.5 * b3 * rsqrt + 2.0 * b4 * rs)
    deps = dq0 * ln - q0 * dq1 / (q1 * (q1 + 1.0))
    return eps, deps


def lda_c_pw92(rho_a, rho_b):
    """PW92 ε_c and v_c^σ."""
    rho_a = np.maximum(np.asarray(rho_a, dtype=float), 1e-30)
    rho_b = np.maximum(np.asarray(rho_b, dtype=float), 1e-30)
    n = rho_a + rho_b
    z = np.clip((rho_a - rho_b) / n, -1 + 1e-12, 1 - 1e-12)
    rs = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)

    def eps_fn(rs, ch):
        e, d = _pw92_eps_depsdrs(rs, **_PW92[ch])
        if ch == "a":
            return -e, -d
        return e, d

    return _spin_interp_vc(rs, z, eps_fn)


# --- PBE exchange vrho (spin-scaled) ----------------------------------------
_MU = 0.2195149727645171
_KAPPA = 0.804


def _pbe_x_unpol(n, gnorm):
    """GGA correction (beyond Slater) for one unpolarized density n, |∇n|."""
    n = np.maximum(n, 1e-30)
    kf = (3.0 * np.pi**2 * n) ** (1.0 / 3.0)
    s = gnorm / (2.0 * kf * n)
    f1 = 1.0 + _MU * s**2 / _KAPPA
    fx1 = _KAPPA - _KAPPA / f1
    exunif = -3.0 * kf / (4.0 * np.pi)
    dsdn = -4.0 / 3.0 * s
    dfxds = 2.0 * _MU * s / f1**2
    vx1 = exunif * fx1 + (exunif / 3.0) * fx1 + exunif * dfxds * dsdn
    return vx1


def pbe_vx(rho_a, rho_b, ga, gb):
    """PBE exchange vrho_σ. ga, gb are |∇ρ_σ|."""
    rho_a = np.maximum(np.asarray(rho_a, dtype=float), 1e-30)
    rho_b = np.maximum(np.asarray(rho_b, dtype=float), 1e-30)
    ga = np.asarray(ga, dtype=float)
    gb = np.asarray(gb, dtype=float)
    vx_a = lda_vx_spin(rho_a) + _pbe_x_unpol(2.0 * rho_a, 2.0 * ga)
    vx_b = lda_vx_spin(rho_b) + _pbe_x_unpol(2.0 * rho_b, 2.0 * gb)
    return vx_a, vx_b


# --- PBE correlation vrho ---------------------------------------------------
_BETA = 0.06672455060314922
_GAMMA = (1.0 - np.log(2.0)) / np.pi**2


def pbe_vc(rho_a, rho_b, gtot):
    """PBE correlation vrho_σ. gtot = |∇(ρ_α+ρ_β)|."""
    rho_a = np.maximum(np.asarray(rho_a, dtype=float), 1e-30)
    rho_b = np.maximum(np.asarray(rho_b, dtype=float), 1e-30)
    n = rho_a + rho_b
    z = np.clip((rho_a - rho_b) / n, -1 + 1e-12, 1 - 1e-12)
    rs = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    ec, vc_a, vc_b = lda_c_pw92(rho_a, rho_b)

    kf = (3.0 * np.pi**2 * n) ** (1.0 / 3.0)
    ks = np.sqrt(4.0 * kf / np.pi)
    phi = 0.5 * ((1 + z) ** (2.0 / 3.0) + (1 - z) ** (2.0 / 3.0))
    phi2, phi3 = phi**2, phi**3
    t = gtot / (2.0 * phi * ks * n)
    t2 = t**2
    expec = np.exp(-ec / (_GAMMA * phi3))
    A = _BETA / (_GAMMA * (expec - 1.0))
    at2 = A * t2
    a2t4 = at2**2
    divsum = 1.0 + at2 + a2t4
    div = (1.0 + at2) / divsum
    nolog = 1.0 + _BETA / _GAMMA * t2 * div
    gec = _GAMMA * phi3 * np.log(nolog)

    dfz = ((1 + z) ** (-1.0 / 3.0) - (1 - z) ** (-1.0 / 3.0)) / 3.0
    dfz = np.nan_to_num(dfz, nan=0.0, posinf=0.0, neginf=0.0)
    factor = a2t4 * (2.0 + at2) / divsum**2
    bfpre = expec / phi3
    dgecpre = _BETA * t2 * phi3 / nolog
    dgec_a = dgecpre * (-7.0 / 3.0 * div - factor * (A * bfpre * (vc_a - ec) / _BETA - 7.0 / 3.0))
    dgec_b = dgecpre * (-7.0 / 3.0 * div - factor * (A * bfpre * (vc_b - ec) / _BETA - 7.0 / 3.0))
    dgeczpre = (
        3.0 * gec / phi
        - _BETA * t2 * phi2 / nolog * (2.0 * div - factor * (3.0 * A * expec * ec / phi3 / _BETA + 2.0))
    ) * dfz
    gvc_a = gec + dgec_a + dgeczpre * (1 - z)
    gvc_b = gec + dgec_b - dgeczpre * (1 + z)
    return vc_a + gvc_a, vc_b + gvc_b


def gga_spin_density(mol, dm, coords):
    ao = numint.eval_ao(mol, coords, deriv=1)
    ru = numint.eval_rho(mol, ao, dm[0], xctype="GGA")
    rd = numint.eval_rho(mol, ao, dm[1], xctype="GGA")
    ga = np.linalg.norm(ru[1:4], axis=0)
    gb = np.linalg.norm(rd[1:4], axis=0)
    gtot = np.linalg.norm(ru[1:4] + rd[1:4], axis=0)
    return ru[0], rd[0], ga, gb, gtot


def analytic_lda(mol, dm, coords):
    ao = numint.eval_ao(mol, coords, deriv=0)
    ra = numint.eval_rho(mol, ao, dm[0], xctype="LDA")
    rb = numint.eval_rho(mol, ao, dm[1], xctype="LDA")
    vx_a, vx_b = lda_vx_spin(ra), lda_vx_spin(rb)
    vc_a, vc_b = lda_vc_vwn(ra, rb)
    return vx_a, vx_b, vx_a + vc_a, vx_b + vc_b


def analytic_pbe(mol, dm, coords):
    """PBE vrho_σ (exchange, and exchange+correlation). Same quantity as libxc[1][0]."""
    ra, rb, ga, gb, gtot = gga_spin_density(mol, dm, coords)
    vx_a, vx_b = pbe_vx(ra, rb, ga, gb)
    vc_a, vc_b = pbe_vc(ra, rb, gtot)
    return vx_a, vx_b, vx_a + vc_a, vx_b + vc_b
