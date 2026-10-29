"""S-II: validate the plaquette winding rule against the closed-form zero lattice
of two Gaussian logons.

s = g(.-t1) e^{2 pi i f1 .} + g(.-t2) e^{2 pi i f2 .} has STFT
V_s(t,f) = sum_j exp(-pi[(t-tj)^2+(f-fj)^2]/2) exp(-i pi (f-fj)(t+tj)).
Zeros lie on the perpendicular bisector of z1 z2, where the phase difference
Delta(t,f) = -pi[(f-f1)(t+t1) - (f-f2)(t+t2)] equals pi mod 2 pi; along the
bisector Delta changes at rate pi d (d = |z1-z2|), so the zeros are spaced 2/d.

For separations d and grid steps delta we compare, inside a disk of radius R
around the midpoint of the pair, the detected plaquettes with the predicted
zeros: counts, and the largest distance from a detected plaquette center to the
nearest predicted zero in units of delta (<= 0.71 when the rule is exact).

Note: the phase of an isolated logon drifts in frequency at rate -pi (t + tj),
whereas the noise field drifts at -2 pi t; after the mixed gauge of (11) the
residual drift of a logon is pi (t - tj), which aliases once |t - tj| delta
exceeds 1/2.  The comparison is therefore reported for R = 3 and R = 5.
"""
import numpy as np
from zerospec.core import detect_zeros


def stft_logons(z1, z2, tg, fg):
    T, F = np.meshgrid(tg, fg)
    V = np.zeros_like(T, dtype=complex)
    for (tj, fj) in (z1, z2):
        V += np.exp(-np.pi * ((T - tj) ** 2 + (F - fj) ** 2) / 2) * \
             np.exp(-1j * np.pi * (F - fj) * (T + tj))
    return V


def predicted_zeros(z1, z2):
    (t1, f1), (t2, f2) = z1, z2
    d = np.hypot(t2 - t1, f2 - f1)
    m = np.array([(t1 + t2) / 2, (f1 + f2) / 2])
    e = np.array([f1 - f2, t2 - t1]) / d
    delta_m = -np.pi * ((m[1] - f1) * (m[0] + t1) - (m[1] - f2) * (m[0] + t2))
    pts = []
    for k in range(-400, 400):
        s = (np.pi - delta_m + 2 * np.pi * k) / (np.pi * d)
        pts.append(m + s * e)
    return np.array(pts), m


def run(R):
    print(f"--- comparison disk radius R = {R} units around the pair midpoint ---")
    print("  d    delta   predicted detected  max|err|/delta")
    for d in (1.0, 2.0, 3.0, 4.0, 6.0):
        for delta in (1 / 16, 1 / 8, 1 / 4):
            ang = 0.4
            z1 = (10.0 - d / 2 * np.cos(ang), 0.0 - d / 2 * np.sin(ang))
            z2 = (10.0 + d / 2 * np.cos(ang), 0.0 + d / 2 * np.sin(ang))
            tg = np.arange(2.0, 18.0 + 1e-9, delta)
            fg = np.arange(-7.0, 7.0 + 1e-9, delta)
            V = stft_logons(z1, z2, tg, fg)
            zs, fc, tc = detect_zeros(V, fg, tg)
            iz, jz = np.nonzero(zs)
            det = np.column_stack([tc[jz], fc[iz]])
            pred, m = predicted_zeros(z1, z2)
            det = det[np.hypot(det[:, 0] - m[0], det[:, 1] - m[1]) <= R]
            pred = pred[np.hypot(pred[:, 0] - m[0], pred[:, 1] - m[1]) <= R]
            if len(pred) and len(det):
                dist = np.sqrt(((det[:, None, :] - pred[None, :, :]) ** 2).sum(-1))
                err = dist.min(axis=1).max() / delta
            else:
                err = np.nan
            print(f"{d:4.1f}  {delta:6.4f}   {len(pred):5d}     {len(det):5d}      {err:6.2f}")


if __name__ == "__main__":
    run(3.0)
    run(5.0)
