"""Shared helpers for the ablations (run_e4_ablations.py).

The masking ablation uses a smooth bimodal spectrum: spectra with sharp,
line-like resonances are bridged by design and belong to the line-handling
path, not to the masking ablation.
"""
import numpy as np
from zerospec.theory import smoothed_psd

DT = 1 / 16

S_smooth = lambda nu: 0.15 + 1.2 * np.exp(-(nu - 0.8) ** 2 / (2 * 0.5 ** 2)) \
                           + 0.7 * np.exp(-(nu + 1.0) ** 2 / (2 * 0.35 ** 2))


def chirp_mix(x, T, rng, gamma0=30.0, S=S_smooth):
    """Add one full-band sweeping chirp of constant local SNR gamma0."""
    tau = np.arange(int(T / DT)) * DT
    finst = -1.7 + (1.7 - (-1.7)) * tau / T
    phase = 2 * np.pi * np.cumsum(finst) * DT
    A = np.sqrt(gamma0 * smoothed_psd(S, finst) / np.sqrt(2.0))
    return x + A * np.exp(1j * (phase + 2 * np.pi * rng.uniform())), finst
