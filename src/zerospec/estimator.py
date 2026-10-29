"""The zero-based spectral estimator (Algorithm 1 of the paper).

Pipeline: STFT -> zero detection -> (optional) masking of signal-occupied
cells by the count-deficit, row, and level tests -> exposure-weighted kernel
intensity -> double integration of 4 pi (rho - 1), with linear bridging across
fully masked columns (1-D harmonic inpainting) -> calibration of the two gauge
constants.  Spectrogram values are used only for the two calibration levels
and the optional level test.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from .core import GaussSTFT, detect_zeros


@dataclass
class ZeroSpecResult:
    f: np.ndarray              # frequency bin centers
    Lambda: np.ndarray         # estimated log smoothed PSD
    rho: np.ndarray            # smoothed zero intensity estimate
    exposure: np.ndarray       # unmasked time measure per column
    bridged: np.ndarray        # columns reconstructed by harmonic bridging
    diagnostics: dict = field(default_factory=dict)


class ZeroSpecEstimator:
    def __init__(self, band=(-2.0, 2.0), margin=1.0, dt=1 / 16, grid=0.125,
                 tstep=2, t_margin=6.0, bandwidth=None, mask=True,
                 cell_f=0.5, cell_t=12.0, alpha=0.008, level_kappa=6.0,
                 dilate=2, calibration="tworef"):
        self.f1, self.f2 = band
        self.margin, self.dt, self.grid, self.tstep = margin, dt, grid, tstep
        self.t_margin, self.bandwidth = t_margin, bandwidth
        self.mask, self.alpha = mask, alpha
        self.cell_f, self.cell_t = cell_f, cell_t
        self.dilate = dilate
        self.level_kappa, self.calibration = level_kappa, calibration

    # main entries
    def fit(self, x: np.ndarray, T: float) -> ZeroSpecResult:
        fgrid = np.arange(self.f1 - self.margin, self.f2 + self.margin + 1e-9, self.grid)
        st = GaussSTFT(T, self.dt)
        V, tgrid = st.stft(x, fgrid, self.tstep)
        return self.fit_stft(V, fgrid, tgrid, T)

    def fit_stft_banded(self, V: np.ndarray, fgrid: np.ndarray, tgrid: np.ndarray,
                        T: float | None = None, chunk: float = 12.0,
                        overlap: float = 2.0) -> ZeroSpecResult:
        """Wide-band variant: split the band into chunks, each self-calibrated
        (two zero-certified references per chunk), and concatenate the chunk
        centers.  Controls the W^2 noise amplification of double integration
        on bands much wider than ~15 units (the paper's multiresolution
        extension, banded-calibration form)."""
        import copy
        f1g, f2g = self.f1, self.f2
        edges = np.arange(f1g, f2g - overlap, chunk - overlap)
        if len(edges) == 0 or edges[-1] + chunk < f2g:
            edges = np.append(edges, f2g - chunk)
        pieces, masks = [], []
        for lo in edges:
            hi = min(lo + chunk, f2g)
            sub = copy.copy(self)
            sub.f1, sub.f2 = lo, hi
            rows = (fgrid >= lo - self.margin - 1e-9) & (fgrid <= hi + self.margin + 1e-9)
            r = sub.fit_stft(V[rows], fgrid[rows], tgrid, T)
            lo_c = lo if lo == edges[0] else lo + overlap / 2
            hi_c = hi if lo == edges[-1] else hi - overlap / 2
            keep = (r.f >= lo_c - 1e-9) & (r.f <= hi_c + 1e-9)
            pieces.append((r, keep))
        f = np.concatenate([r.f[k] for r, k in pieces])
        order = np.argsort(f)
        cat = lambda attr: np.concatenate(
            [getattr(r, attr)[k] for r, k in pieces])[order]
        return ZeroSpecResult(f=f[order], Lambda=cat("Lambda"), rho=cat("rho"),
                              exposure=cat("exposure"), bridged=cat("bridged"),
                              diagnostics=dict(pieces=[r.diagnostics for r, _ in pieces]))

    def fit_stft(self, V: np.ndarray, fgrid: np.ndarray, tgrid: np.ndarray,
                 T: float | None = None) -> ZeroSpecResult:
        """Run the pipeline on a precomputed STFT (normalized coordinates)."""
        T = T if T is not None else float(tgrid[-1])
        zs, fc, tc = detect_zeros(V, fgrid, tgrid)
        keep_t = (tc >= tgrid[0] + self.t_margin) & (tc <= T - self.t_margin)
        zs, tc = zs[:, keep_t], tc[keep_t]
        Vp = V[:, :-1][:, keep_t]                       # plaquette-aligned values
        # |V|^2 at plaquette centers: average magnitudes (adjacent rows carry a
        # deterministic phase drift -2 pi t df, so complex averaging is invalid)
        P2 = 0.5 * (np.abs(Vp[:-1, :]) ** 2 + np.abs(Vp[1:, :]) ** 2)

        dtp = tc[1] - tc[0]
        dfp = fc[1] - fc[0]

        # masking
        masked = np.zeros_like(zs, dtype=bool)
        if self.mask:
            masked = self._mask_cells(zs, P2, fc, tc, dtp, dfp)

        # exposure-weighted column densities
        unm = ~masked
        exposure = unm.sum(axis=1) * dtp                 # time measure per column
        counts = (zs & unm).sum(axis=1).astype(float)
        # counts/exposure = zeros per unit time in a column of width dfp,
        # so dividing by dfp gives zeros per unit TF area:
        rho_col = counts / np.maximum(exposure, 1e-12) / dfp

        good = exposure > 0.3 * exposure.max()

        # kernel smoothing (exposure-weighted NW)
        # bandwidth rule b = c T^{-1/5}; c = 0.7 from pilot runs on the test
        # spectra family (balances peak-curvature bias against count noise)
        b = self.bandwidth or max(0.7 * (tc[-1] - tc[0]) ** (-1 / 5), 1.2 * dfp)
        D = fc[:, None] - fc[None, :]
        K = np.where(np.abs(D) <= 3 * b, np.exp(-0.5 * (D / b) ** 2), 0.0)
        w = np.where(good, exposure, 0.0)
        num = K @ (w * rho_col)
        den = K @ w
        den_typ = np.median(den[den > 0]) if np.any(den > 0) else 1.0
        bridged = den < 0.05 * den_typ
        rho_s = np.where(~bridged, num / np.maximum(den, 1e-300), 1.0)
        # bridged columns: Lambda'' = 0 there => linear (harmonic) inpainting

        # double integration
        rhs = 4 * np.pi * (rho_s - 1.0)
        Lp = np.concatenate([[0.0], np.cumsum(0.5 * (rhs[1:] + rhs[:-1]) * np.diff(fc))])
        Lam = np.concatenate([[0.0], np.cumsum(0.5 * (Lp[1:] + Lp[:-1]) * np.diff(fc))])

        # gauge calibration
        Lam = self._calibrate(Lam, fc, P2, masked, good, tc)

        inband = (fc >= self.f1) & (fc <= self.f2)
        return ZeroSpecResult(
            f=fc[inband], Lambda=Lam[inband], rho=rho_s[inband],
            exposure=exposure[inband], bridged=bridged[inband],
            diagnostics=dict(zs=zs, masked=masked, fc=fc, tc=tc, P2=P2,
                             rho_col=rho_col, good=good, b=b, Lam_full=Lam))

    # helpers
    def _mask_cells(self, zs, P2, fc, tc, dtp, dfp):
        from scipy.stats import poisson
        from scipy.ndimage import binary_dilation
        # cells elongated in time (cell_f x cell_t units): a signal's zero-free
        # tube is ~1 unit wide in f and locally horizontal, so thin-in-f cells
        # see a total zero deficit while plain noise keeps mu ~ cell_f*cell_t
        nfb = max(1, int(round(self.cell_f / dfp)))
        ntb = max(1, int(round(self.cell_t / dtp)))
        nF = int(np.ceil(zs.shape[0] / nfb))
        nT = int(np.ceil(zs.shape[1] / ntb))
        cnt = np.zeros((nF, nT))
        lev = np.zeros((nF, nT))
        for i in range(nF):
            for j in range(nT):
                blk = np.s_[i * nfb:(i + 1) * nfb, j * ntb:(j + 1) * ntb]
                cnt[i, j] = zs[blk].sum()
                lev[i, j] = np.median(P2[blk])
        area = nfb * dfp * ntb * dtp
        rho_pilot = max(zs.mean() / (dfp * dtp), 0.2)
        mu = rho_pilot * area
        # largest k with P(N<=k) <= alpha (Poisson is conservative for the
        # hyperuniform background); -1 -> never flag by count
        kthr = -1
        while poisson.cdf(kthr + 1, mu) <= self.alpha:
            kthr += 1
        flag_zero = cnt <= kthr
        # persistent-deficit row test, relative to neighboring rows: a tone/line
        # depletes one or two rows while adjacent rows stay normal, whereas a
        # smooth spectral peak depresses a whole band of rows together (and a
        # genuine noise resonance must not be masked).  Reference mean = 75th
        # percentile of rows within +-6 blocks (excluding immediate
        # neighbors); flag on a deep Poisson tail plus an effect-size guard.
        row_tot = cnt.sum(axis=1)
        nF_ = len(row_tot)
        flag_row = np.zeros(nF_, dtype=bool)
        for i in range(nF_):
            nbr = np.concatenate([row_tot[max(0, i - 6):max(0, i - 1)],
                                  row_tot[i + 2:i + 8]])
            if nbr.size < 3:
                continue
            ref = np.percentile(nbr, 75)
            if ref <= 0:
                continue
            if row_tot[i] <= 0.6 * ref and poisson.cdf(row_tot[i], ref) <= 1e-4:
                flag_row[i] = True
        # level pilot per frequency block: median over time blocks at the same
        # f (transient/strong dwellers exceed it; a persistent tone equals its
        # own pilot and is caught instead by the row deficit test; genuine
        # spectral peaks are flat in time and never flagged)
        pil = np.median(lev, axis=1)
        flag_level = lev > self.level_kappa * pil[:, None]
        flag_zero = flag_zero | flag_row[:, None]
        # dilation reach must cover the enhancement ring around strong signals
        # (rho stays >1.05 out to ~1.2 units beyond the dead core)
        flags = binary_dilation(flag_zero | flag_level, iterations=self.dilate)
        masked = np.zeros_like(zs, dtype=bool)
        for i in range(nF):
            for j in range(nT):
                if flags[i, j]:
                    masked[i * nfb:(i + 1) * nfb, j * ntb:(j + 1) * ntb] = True
        return masked

    def _quantile_level(self, P2, masked, col, q=0.25):
        vals = P2[col][~masked[col]]
        if vals.size < 20:
            return None
        return np.quantile(vals, q) / (-np.log1p(-q))

    def _calibrate(self, Lam, fc, P2, masked, good, tc):
        idx = np.where(good)[0]
        if idx.size < 2:                      # fallback: least-masked columns
            expo = (~masked).sum(axis=1)
            idx = np.argsort(expo)[-max(4, len(expo) // 10):]
            idx = np.sort(idx)
        if self.calibration == "tworef":
            # two robust references near the band edges, on well-exposed columns
            lo = idx[idx <= np.quantile(idx, 0.25)]
            hi = idx[idx >= np.quantile(idx, 0.75)]
            cands = [int(lo[len(lo) // 2]), int(hi[len(hi) // 2])]
            refs = []
            for c in cands:
                p = self._quantile_level(P2, masked, c)
                if p is not None and p > 0:
                    refs.append((fc[c], np.log(p), c))
            if len(refs) == 2 and abs(refs[1][0] - refs[0][0]) > 1e-9:
                (fa, la, ca), (fb, lb, cb) = refs
                a1 = ((lb - Lam[cb]) - (la - Lam[ca])) / (fb - fa)
                a0 = (la - Lam[ca]) - a1 * fa
                return Lam + a0 + a1 * fc
            if len(refs) == 1:                # level-only fallback
                fa, la, ca = refs[0]
                return Lam + (la - Lam[ca])
            # unconditional fallback: one global robust level anchor (a1 = 0)
            vals = P2[~masked]
            if vals.size > 50:
                anchor = np.log(np.quantile(vals, 0.25) / (-np.log1p(-0.25)))
                mid = len(Lam) // 2
                return Lam + (anchor - Lam[mid])
            return Lam
        raise ValueError(self.calibration)
