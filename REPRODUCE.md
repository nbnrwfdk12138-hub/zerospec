# Reproducing the figures and tables

Code to reproduce every figure and table of the paper "Spectrogram Zeros of Colored Noise:
An Intensity Formula and Spectral Estimation Without Spectral Averaging" and of its
supplementary material (SM).

## Environment

- Python 3.14, managed with [`uv`](https://docs.astral.sh/uv/). Install with `uv sync`.
- Run any script with `uv run python scripts/<script>.py`.
- Core package `src/zerospec/`: the mixed-gauge zero detector (rule (11) of the paper),
  the estimator of Algorithm 1 (kernel-smoothed zero counting, masking tests, Poisson
  inversion, two-reference calibration, subband calibration), the Welch /
  minimum-statistics / MMSE-SPP baselines (Welch is convolved with the spectral window h,
  `baselines.welch_h`), and the figure style.

## Data (all public)

| Dataset | Access |
|---|---|
| GW150914 strain (H1, 4 kHz) | GWOSC, gwosc.org |
| LibriSpeech `test-clean` | openslr.org/resources/12 |
| DEMAND (DKITCHEN, PCAFETER) | Zenodo 10.5281/zenodo.1227121 |
| U. Ottawa variable-speed bearing | Mendeley v43hmbwxpm; save under `data/ottawa/` |

Run `uv run python scripts/download_data.py` first (Ottawa is downloaded manually).

## Artifact -> experiment script -> figure script

Run the experiment script first (it writes `results/*.npz`), then the figure scripts.
Main-text figures are three-panel rows written by `make_rows3_r1.py` to
`figures/rows3_r1/`; SM figures are written by `make_sm_panels_r1.py` to the same folder.

| Figure or table | Experiment script | Figure scripts |
|---|---|---|
| Fig. 1 (intensity laws) | `run_e1_intensity.py` | `make_fig1.py`, `make_panels_r1.py`, `make_rows3_r1.py fig1_row3` |
| Fig. 2 (clean noise, rate) | `run_e2_estimator.py` | `make_fig2.py`, `make_panels_r1.py`, `make_rows3_r1.py fig2_row3` |
| Fig. 3 (strong in-band signals) | `run_e3_robustness.py` | `make_fig3.py`, `make_panels_r1.py`, `make_rows3_r1.py fig3_row3` |
| Fig. 4 (LIGO, GW150914) | `run_e3b_ligo.py` | `make_fig4.py`, `make_panels_r1.py`, `make_rows3_r1.py fig4_row3` |
| Fig. 5 (speech) | `run_e3c_speech.py` | `make_fig5.py`, `make_rows3_r1.py fig5_row3` |
| SM Fig. 6 (detector, white-noise density) | `run_e4_ablations.py` | `make_fig6.py`, `make_sm_panels_r1.py figS6_c` |
| SM Fig. 7 (ablations) | `run_e4_ablations.py` | `make_fig6.py`, `make_sm_panels_r1.py figS7_row` |
| SM Fig. 8 (bearing) | `run_e3d_bearing.py H-A-1`, then `run_e3d_bearing.py I-A-1` | `make_fig7.py`, `make_sm_panels_r1.py figS8_row` |
| SM Fig. 9 (real white noise near DC) | `run_s5_realnoise.py` | `make_figS.py`, `make_sm_panels_r1.py figS9_c` |
| SM Table II (enhancement) | `run_e3c_enhance.py` | values printed by the script |
| SM Sec. S-II (two-logon zero lattice) | `run_s2_lattice.py` | table printed by the script |
| SM Sec. S-III-A (detection rates, runtime) | `run_e5_roc.py` | values printed by the script |

Every run fixes the seed of NumPy's default generator in its script; the seeds are also
listed in SM Sec. S-IV. Default analysis parameters are those of Table I of the paper; the
four departures from them are stated in Section VIII.
