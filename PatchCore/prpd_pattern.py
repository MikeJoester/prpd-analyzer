"""phi-q-n PRPD patterns: the conventional 2D view of a PD measurement (Analysis J / L).

Our raw file is `128 phase bins x 3600 cycles` of peak amplitudes — a phase-resolved *pulse sequence*
(PRPS). The representation used in the PD literature is the **phi-q-n plot**: phase on x, amplitude
(apparent charge) on y, colour = number of pulses. It is obtained by histogramming amplitudes over
the cycle axis, so no extra data is needed.

Also provides the classic phi-q-n statistical descriptors per half cycle (pulse count, mean/max
amplitude, skewness, kurtosis, phase asymmetry, cross-correlation between the two half cycles).
"""

from __future__ import annotations

import numpy as np

PHASE_BINS, AMPLITUDE_BINS = 128, 256


def phi_q_n(matrix: np.ndarray) -> np.ndarray:
    """(128 phase, T cycles) uint8 -> (256 amplitude, 128 phase) pulse-count histogram."""
    h = np.zeros((AMPLITUDE_BINS, PHASE_BINS), dtype=np.int32)
    for p in range(matrix.shape[0]):
        v = matrix[p]
        v = v[v > 0]
        if v.size:
            h[:, p] = np.bincount(v, minlength=AMPLITUDE_BINS)
    return h


def plot_phi_q_n(ax, matrix: np.ndarray, title: str = "", cmap: str = "viridis") -> None:
    """Draw the phi-q-n pattern with a log count scale (pulse counts span orders of magnitude)."""
    h = phi_q_n(matrix)
    ax.imshow(np.log1p(h), origin="lower", aspect="auto", cmap=cmap,
              extent=(0, 360, 0, AMPLITUDE_BINS - 1), interpolation="nearest")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("phase (deg)", fontsize=8)
    ax.set_ylabel("amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=8.5)


def _moments(weights: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Weighted skewness and kurtosis of distribution `weights` over positions `x`."""
    w = weights.astype(float)
    if w.sum() <= 0:
        return float("nan"), float("nan")
    mu = np.average(x, weights=w)
    var = np.average((x - mu) ** 2, weights=w)
    if var <= 0:
        return float("nan"), float("nan")
    sk = np.average((x - mu) ** 3, weights=w) / var ** 1.5
    ku = np.average((x - mu) ** 4, weights=w) / var ** 2 - 3.0
    return float(sk), float(ku)


def descriptors(matrix: np.ndarray) -> dict:
    """Classic phi-q-n descriptors, computed per half cycle (phase bins 0-63 and 64-127)."""
    h = phi_q_n(matrix)
    amps = np.arange(AMPLITUDE_BINS)
    out: dict[str, float] = {}
    profiles = {}
    for name, sl in (("pos", slice(0, PHASE_BINS // 2)), ("neg", slice(PHASE_BINS // 2, PHASE_BINS))):
        part = h[:, sl]
        n = int(part.sum())
        counts_by_phase = part.sum(axis=0)
        profiles[name] = counts_by_phase.astype(float)
        amp_dist = part.sum(axis=1)
        sk, ku = _moments(amp_dist, amps)
        mean_amp = float(np.average(amps, weights=amp_dist)) if n else float("nan")
        nz = np.nonzero(amp_dist)[0]
        out |= {
            f"{name}_pulse_count": n,
            f"{name}_mean_amplitude": mean_amp,
            f"{name}_max_amplitude": float(nz.max()) if nz.size else float("nan"),
            f"{name}_amp_skewness": sk,
            f"{name}_amp_kurtosis": ku,
            f"{name}_active_phase_bins": int((counts_by_phase > 0).sum()),
        }
    # asymmetry and cross-correlation between the two half cycles (standard PD operators)
    a, b = profiles["pos"], profiles["neg"]
    out["count_asymmetry"] = float((a.sum() - b.sum()) / (a.sum() + b.sum())) if (a.sum() + b.sum()) else float("nan")
    out["amplitude_asymmetry"] = (float(out["neg_mean_amplitude"] / out["pos_mean_amplitude"])
                                  if out["pos_mean_amplitude"] else float("nan"))
    if a.std() > 0 and b.std() > 0:
        out["half_cycle_cross_correlation"] = float(np.corrcoef(a, b)[0, 1])
    else:
        out["half_cycle_cross_correlation"] = float("nan")
    out["total_pulses"] = int(h.sum())
    out["occupied_cells"] = int((h > 0).sum())
    return out
