"""zerospec: spectrogram-zero based noise spectral estimation.

Window-normalized coordinates throughout: unit Gaussian window
g(tau) = 2^{1/4} exp(-pi tau^2), |ghat|^2 of unit mass and std 1/(2 sqrt(pi)).
"""
from .core import GaussSTFT, detect_zeros, synth_stationary, synth_modulated
from .theory import smoothed_psd, rho_stationary, rho_modulated, rho_noncentral_tone
from .estimator import ZeroSpecEstimator
from . import baselines, figstyle
