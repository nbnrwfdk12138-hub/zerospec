"""Theory-side formulas: smoothed spectra and zero intensities."""
from __future__ import annotations
import numpy as np

# |ghat|^2(xi) = sqrt(2) exp(-2 pi xi^2): unit mass, std 1/(2 sqrt(pi))
GHAT2 = lambda xi: np.sqrt(2.0) * np.exp(-2.0 * np.pi * xi ** 2)
GABS2 = GHAT2  # |g|^2 has the same shape in time


def smoothed_psd(Sfunc, fline: np.ndarray, half: float = 4.0, step: float = 1e-3):
    """P(f) = (S * |ghat|^2)(f) by quadrature."""
    xi = np.arange(-half, half, step)
    w = GHAT2(xi) * step
    return np.array([np.sum(Sfunc(f - xi) * w) for f in np.atleast_1d(fline)])


def rho_stationary(Sfunc, fline: np.ndarray):
    """rho(f) = 1 + (log P)''(f) / 4pi for stationary noise."""
    lp = np.log(smoothed_psd(Sfunc, fline))
    return 1.0 + np.gradient(np.gradient(lp, fline), fline) / (4 * np.pi)


def rho_modulated(sig2_func, tline: np.ndarray, half: float = 4.0, step: float = 1e-3):
    """rho(t) = 1 + (d^2/dt^2) log(sig^2 * |g|^2)(t) / 4pi for modulated white noise."""
    s = np.arange(-half, half, step)
    w = GABS2(s) * step
    lp = np.log(np.array([np.sum(sig2_func(t - s) * w) for t in np.atleast_1d(tline)]))
    return 1.0 + np.gradient(np.gradient(lp, tline), tline) / (4 * np.pi)


def rho_noncentral_tone(fline: np.ndarray, A: float, f0: float = 0.0, P: float = 1.0):
    """Exact non-central intensity for white noise (PSD P) + tone A e^{2i pi f0 t}:
    rho = e^{-gam} (1 + 4 pi (f-f0)^2 gam),  gam = A^2 sqrt(2) e^{-2 pi (f-f0)^2} / P.
    """
    d = np.atleast_1d(fline) - f0
    gam = A ** 2 * np.sqrt(2.0) * np.exp(-2.0 * np.pi * d ** 2) / P
    return np.exp(-gam) * (1.0 + 4 * np.pi * d ** 2 * gam)


def bin_average(func, centers: np.ndarray, width: float, nsub: int = 17):
    """Average a callable over bins (the comparator for binned counts)."""
    sub = np.linspace(-width / 2, width / 2, nsub)
    vals = func(centers[:, None] + sub[None, :])
    return vals.mean(axis=1)
