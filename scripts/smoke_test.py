import numpy as np
from zerospec import GaussSTFT, detect_zeros, synth_stationary, ZeroSpecEstimator
from zerospec.theory import smoothed_psd

rng = np.random.default_rng(3)
T = 256.0
S = lambda nu: 0.15 + 1.2 * np.exp(-(nu - 0.8) ** 2 / (2 * 0.5 ** 2)) \
             + 0.7 * np.exp(-(nu + 1.0) ** 2 / (2 * 0.35 ** 2))

# 1) detector check on white noise
x = synth_stationary(lambda nu: np.ones_like(np.atleast_1d(nu), float), T, rng)
st = GaussSTFT(T)
fg = np.arange(-2.5, 2.5 + 1e-9, 0.125)
V, tg = st.stft(x, fg)
zs, fc, tc = detect_zeros(V, fg, tg)
sel = (tc >= 6) & (tc <= T - 6)
dens = zs[:, sel].sum() / ((fg[-1] - fg[0] - 0.125) * (tc[sel][-1] - tc[sel][0]))
print(f"white density = {dens:.4f} (expect ~1)")

# 2) estimator end-to-end on colored noise, one realization, no masking
x = synth_stationary(S, T, rng)
est = ZeroSpecEstimator(band=(-1.8, 1.8), mask=False)
r = est.fit(x, T)
truth = np.log(smoothed_psd(S, r.f))
err = r.Lambda - truth
print(f"[no mask]  sup|err| = {np.abs(err).max():.3f}  rms = {np.sqrt((err**2).mean()):.3f}  "
      f"bridged={r.bridged.sum()}  b={r.diagnostics['b']:.3f}")

# 3) same with masking enabled (should not hurt on clean noise)
est2 = ZeroSpecEstimator(band=(-1.8, 1.8), mask=True)
r2 = est2.fit(x, T)
err2 = r2.Lambda - np.log(smoothed_psd(S, r2.f))
print(f"[masking]  sup|err| = {np.abs(err2).max():.3f}  rms = {np.sqrt((err2**2).mean()):.3f}  "
      f"masked_frac = {r2.diagnostics['masked'].mean():.3f}")
