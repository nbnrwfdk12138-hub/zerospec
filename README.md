# Spectrogram Zeros of Colored Noise

Code and stored results for the paper *Spectrogram Zeros of Colored Noise: An Intensity
Formula and Spectral Estimation Without Spectral Averaging* (submitted to IEEE Transactions
on Signal Processing).

The method estimates the window-smoothed noise power spectral density from the zeros of a
Gaussian-window spectrogram. It detects the zeros on a grid, masks the cells that a strong
signal empties, smooths the zero counts, and inverts the intensity law by double
integration; two reference levels fix the remaining affine gauge. All quantities are in
window-normalized units, with the unit Gaussian window g(t) = 2^(1/4) exp(-pi t^2).

## Contents

| Path | Content |
|---|---|
| `src/zerospec/` | The package: Gaussian-window STFT and zero detector, rule (11) of the paper (`core.py`); intensity formulas (`theory.py`); the estimator of Algorithm 1 (`estimator.py`); the Welch, minimum-statistics, and MMSE-SPP baselines (`baselines.py`); figure style (`figstyle.py`) |
| `scripts/run_*.py` | Experiments; each writes `results/*.npz` |
| `scripts/make_*.py` | Figures; they read `results/*.npz` and write to `figures/` |
| `results/` | The stored results behind every figure and table of the paper and its supplementary material (SM) |
| `REPRODUCE.md` | Which script produces which figure or table |

## Installation

Python 3.12 or later; the results were produced with Python 3.14. With
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python scripts/smoke_test.py
```

The smoke test runs in a few seconds. It prints the detected zero density of white noise,
which should be close to 1, and the estimation error on one colored-noise record. Without
uv, `pip install -e .` installs the package and its dependencies.

## Regenerating the figures from the stored results

```bash
uv run python scripts/make_rows3_r1.py      # Figs. 1-5 of the paper
uv run python scripts/make_sm_panels_r1.py  # Figs. 6-9 of the SM
```

Both write PDF and PNG files to `figures/rows3_r1/`. Most numbers quoted in the paper are
stored in the same `results/*.npz` files; `REPRODUCE.md` marks the few that a script prints
instead.

## Rerunning the experiments

Download the data first (about 2 GB in all):

```bash
uv run python scripts/download_data.py
```

This fetches the LIGO, LibriSpeech, and DEMAND data and unpacks the archives where the
experiments read them. The Ottawa bearing data may have to be downloaded by hand from
Mendeley Data (see below) and unpacked so that
`data/ottawa/v43hmbwxpm-2/1 Data collected from a healthy bearing/H-A-1.mat` exists.

Then run the experiment scripts with `uv run python scripts/<script>`:

| Script | Paper |
|---|---|
| `run_e1_intensity.py` | Fig. 1 |
| `run_e2_estimator.py` | Fig. 2 |
| `run_e3_robustness.py` | Fig. 3 |
| `run_e3b_ligo.py` | Fig. 4 |
| `run_e3c_speech.py` | Fig. 5 |
| `run_e3c_enhance.py` | SM Table II |
| `run_e3d_bearing.py H-A-1`, then `run_e3d_bearing.py I-A-1` | SM Fig. 8 |
| `run_e4_ablations.py` | SM Figs. 6 and 7 |
| `run_e5_roc.py` | SM Sec. S-III-A (detection rates, runtime) |
| `run_s2_lattice.py` | SM Sec. S-II (zero lattice of two logons) |
| `run_s5_realnoise.py` | SM Fig. 9 |

On a desktop PC the runs take from about a minute (bearing) to about an hour (Fig. 3).
Each script fixes its random seed; the seeds are also listed in SM Sec. S-IV. Rerunning
reproduces the stored results up to floating-point differences between platforms and
library versions.

## Data

The data are public and are not redistributed here.

| Dataset | Source | Used in |
|---|---|---|
| GW150914 strain, H1, 4 kHz | GWOSC, https://gwosc.org | Fig. 4 |
| LibriSpeech `test-clean` | https://www.openslr.org/12 | Fig. 5, SM Table II |
| DEMAND, environments DKITCHEN and PCAFETER | Zenodo, https://doi.org/10.5281/zenodo.1227121 | Fig. 5, SM Table II |
| University of Ottawa variable-speed bearing data | Mendeley Data, https://data.mendeley.com/datasets/v43hmbwxpm/2 | SM Fig. 8 |

## License

The code is released under the MIT License (see `LICENSE`). The datasets keep the licenses
of their providers.

## Citation

If you use this code, please cite the paper. The full reference will be added here on
publication.
