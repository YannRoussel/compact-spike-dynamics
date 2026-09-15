"""Time-resolved loop features and a long-square, event-based adaptation clock.

Time is measured from the first spike. This is a protocol-conditioned model,
not an autonomous recovery-state model for arbitrary injected currents.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .phase_template_model import PhaseTemplateSimulation
from .phase_timing import _periodic_spline
from .recovery_onset import _normalized_exponential


SEQUENCE_FEATURES = (
    "isi_ms", "loop_area_mv2_ms", "amplitude_mv", "max_upstroke_mv_ms",
    "downstroke_magnitude_mv_ms",
)
TAUS_MS = np.array([20.0, 100.0, 500.0])
FRACTIONS = np.linspace(0.0, 1.0, 5)


def extract_time_phase_features(trace):
    """Extract every complete peak-to-peak loop from an already smoothed trace.

    Acceleration roots are interpolated sign changes in d2V/dt2. They are
    temporal inflections, not geometric inflections of the (V, dV/dt) loop.
    The 3D arc uses fixed scales: 100 ms, 100 mV, 100 mV/ms.
    """
    t, v, u = trace.time_ms, trace.voltage_mv, trace.velocity_mv_ms
    dt = float(np.median(np.diff(t)))
    active = np.flatnonzero((t >= trace.stimulus_start_ms) &
                           (t < trace.stimulus_end_ms))
    peaks = active[find_peaks(v[active], height=0.0,
                             distance=max(1, int(round(1.0 / dt))))[0]]
    rows = []
    for number, (start, stop) in enumerate(zip(peaks[:-1], peaks[1:])):
        sl = slice(start, stop + 1)
        tt, vv, uu = t[sl], v[sl], u[sl]
        acceleration = np.gradient(uu, tt)
        nonzero = np.flatnonzero(acceleration != 0)
        roots = [(j, k) for j, k in zip(nonzero[:-1], nonzero[1:])
                 if acceleration[j] * acceleration[k] < 0]
        root_v, root_u = [], []
        for j, k in roots:
            f = -acceleration[j] / (acceleration[k] - acceleration[j])
            root_v.append(float(vv[j] + f * (vv[k] - vv[j])))
            root_u.append(float(uu[j] + f * (uu[k] - uu[j])))
        xyz = np.column_stack((tt / 100.0, vv / 100.0, uu / 100.0))
        area = abs(0.5 * np.sum(vv * np.roll(uu, -1) - uu * np.roll(vv, -1)))
        rows.append({
            "cycle_index": number, "elapsed_ms": float(tt[0] - t[peaks[0]]),
            "peak_time_ms": float(tt[0]), "isi_ms": float(tt[-1] - tt[0]),
            "loop_area_mv2_ms": float(area), "amplitude_mv": float(np.ptp(vv)),
            "max_upstroke_mv_ms": float(np.max(uu)),
            "downstroke_magnitude_mv_ms": float(-np.min(uu)),
            "peak_mv": float(np.max(vv)), "trough_mv": float(np.min(vv)),
            "voltage_at_max_upstroke_mv": float(vv[np.argmax(uu)]),
            "voltage_at_max_downstroke_mv": float(vv[np.argmin(uu)]),
            "arc_length_3d": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
            "acceleration_root_count": len(roots),
            "acceleration_root_voltage_mv": root_v,
            "acceleration_root_velocity_mv_ms": root_u,
            "current_pa": float(np.median(trace.input_value[active])),
            "trace_name": trace.name,
        })
    return pd.DataFrame(rows)


def temporal_basis(elapsed_ms):
    t = np.asarray(elapsed_ms, dtype=float).reshape(-1)
    if not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("Elapsed time must be finite and nonnegative")
    return np.column_stack((np.ones(len(t)), -np.expm1(-t[:, None] / TAUS_MS)))


@dataclass(frozen=True)
class SequenceClock:
    currents: np.ndarray
    coefficients: np.ndarray  # current x basis x output, in log physical units

    def predict(self, current, elapsed_ms):
        coefficients = np.array([
            np.interp(current, self.currents, self.coefficients[:, b, j])
            for b in range(4) for j in range(self.coefficients.shape[2])
        ]).reshape(4, -1)
        return np.exp(np.clip(temporal_basis(elapsed_ms) @ coefficients, -8, 16))

    def period(self, current, elapsed_ms):
        return float(np.clip(self.predict(current, [elapsed_ms])[0, 0], 1.0, 2000.0))


def _fit_clock(tables, ridge, rank, joint):
    names = SEQUENCE_FEATURES if joint else SEQUENCE_FEATURES[:1]
    levels, coefficients = [], []
    for table in tables:
        x = temporal_basis(table.elapsed_ms.to_numpy())
        y = np.log(np.maximum(table[list(names)].to_numpy(float), 1e-6))
        center = y.mean(axis=0)
        scale = np.maximum(y.std(axis=0), 0.05)
        z = (y - center) / scale
        penalty = np.diag([0.0, ridge, ridge, ridge])
        coef = np.linalg.solve(x.T @ x / len(x) + penalty,
                               x.T @ z / len(x))
        # Project the fitted multi-output trajectory onto shared response axes.
        _, _, vt = np.linalg.svd(x @ coef, full_matrices=False)
        projection = vt[:rank].T @ vt[:rank]
        coef = (coef @ projection) * scale
        coef[0] += center
        levels.append(float(table.current_pa.iloc[0]))
        coefficients.append(coef)
    order = np.argsort(levels)
    return SequenceClock(np.asarray(levels)[order], np.asarray(coefficients)[order])


def fit_sequence_clock(tables, joint=False):
    """Select complexity on late training cycles using free-running event times."""
    tables = [table for table in tables if len(table) >= 3]
    if not tables:
        raise ValueError("No training trace with at least three complete cycles")
    splits = [max(2, int(0.7 * len(table))) for table in tables]
    candidates = []
    for ridge in (0.001, 0.01, 0.1, 1.0):
        for rank in ((1, 2, 5) if joint else (1,)):
            model = _fit_clock([t.iloc[:n] for t, n in zip(tables, splits)],
                               ridge, rank, joint)
            scores = []
            for table, n in zip(tables, splits):
                elapsed, predictions = 0.0, []
                for _ in range(len(table)):
                    pred = model.predict(float(table.current_pa.iloc[0]), [elapsed])[0]
                    predictions.append(pred)
                    elapsed += float(np.clip(pred[0], 1.0, 2000.0))
                names = SEQUENCE_FEATURES if joint else SEQUENCE_FEATURES[:1]
                observed = np.log(np.maximum(table[list(names)].to_numpy(float), 1e-6))
                error = np.log(np.maximum(predictions, 1e-6)) - observed
                score = np.mean(error[n:, 0] ** 2)
                if joint:
                    score += 0.2 * np.mean(error[n:, 1:] ** 2)
                scores.append(score)
            candidates.append((float(np.mean(scores)), ridge, rank))
    score, ridge, rank = min(candidates)
    return _fit_clock(tables, ridge, rank, joint), {
        "ridge": ridge, "rank": rank, "inner_score": score,
    }


def clock_parameter_row(clock, low, high):
    """Export timing coordinates on the same five current fractions as waveform."""
    return {
        f"sequence_logperiod_q{int(q * 100):03d}_b{b}": float(
            np.interp(low + q * (high - low), clock.currents, clock.coefficients[:, b, 0])
        ) for q in FRACTIONS for b in range(4)
    }


def clock_from_row(row):
    low, span = float(row["repetitive_current_min_pa"]), float(row["repetitive_current_span_pa"])
    coefficients = np.array([
        [row[f"sequence_logperiod_q{int(q * 100):03d}_b{b}"] for b in range(4)]
        for q in FRACTIONS
    ], dtype=float)[:, :, None]
    return SequenceClock(low + FRACTIONS * max(span, 1e-6), coefficients)


def simulate_sequence(waveform, onset, clock, trace):
    """Render a freely generated event sequence; observed spike times are unused."""
    t = trace.time_ms
    current = float(np.median(trace.input_value[(t >= trace.stimulus_start_ms) &
                                               (t < trace.stimulus_end_ms)]))
    v = np.full_like(t, waveform.resting_voltage_mv)
    phase = np.zeros_like(t)
    periods = np.full_like(t, np.nan)
    spline = _periodic_spline(waveform, current)
    down, _, up = waveform.segment_durations(current, clock.period(current, 0))
    first = trace.stimulus_start_ms + max(up + 0.1, waveform.latency(current))
    onset_stop = first - up
    mask = (t >= trace.stimulus_start_ms) & (t < onset_stop)
    v[mask] += _normalized_exponential(t[mask] - trace.stimulus_start_ms,
                                     onset_stop - trace.stimulus_start_ms,
                                     onset.tau(current)) * (float(spline(0.75)) - v[mask])
    mask = (t >= onset_stop) & (t < first) & (t < trace.stimulus_end_ms)
    phase[mask] = 0.75 + 0.25 * (t[mask] - onset_stop) / up
    v[mask] = spline(phase[mask])
    peak = first
    while peak < trace.stimulus_end_ms:
        period = max(down + up + 0.1, clock.period(current, peak - first))
        mask = (t >= peak) & (t < peak + period) & (t < trace.stimulus_end_ms)
        phase[mask] = np.interp(t[mask] - peak, [0, down, period - up, period],
                               [0, 0.25, 0.75, 1])
        v[mask] = spline(phase[mask])
        periods[mask] = period
        peak += period
    after = np.flatnonzero(t >= trace.stimulus_end_ms)
    if len(after) and after[0] > 0:
        v[after] += (v[after[0] - 1] - waveform.resting_voltage_mv) * np.exp(
            -(t[after] - trace.stimulus_end_ms) / waveform.rest_relaxation_ms)
    return PhaseTemplateSimulation(t, v, np.gradient(v, t), phase,
                                   np.zeros_like(t), periods)


def train_metrics(trace, voltage):
    """Timing metrics with explicit counts; missing ISIs stay missing, not zero."""
    t = trace.time_ms
    active = (t >= trace.stimulus_start_ms) & (t < trace.stimulus_end_ms)
    def spikes(v):
        return t[active][find_peaks(np.asarray(v)[active], height=0.0,
                                  distance=max(1, int(round(1 / np.median(np.diff(t))))))[0]]
    a, b = spikes(trace.voltage_mv), spikes(voltage)
    ia, ib = np.diff(a), np.diff(b)
    n = min(len(ia), len(ib))
    ratio = lambda isi: float(np.mean(isi[-3:]) / np.mean(isi[:3])) if len(isi) >= 3 else np.nan
    return {
        "observed_spike_count": len(a), "predicted_spike_count": len(b),
        "count_error": abs(len(a) - len(b)), "matched_isi_count": n,
        "isi_rmse_ms": float(np.sqrt(np.mean((ia[:n] - ib[:n]) ** 2))) if n else np.nan,
        "early_isi_rmse_ms": float(np.sqrt(np.mean((ia[:min(3,n)] - ib[:min(3,n)]) ** 2))) if n else np.nan,
        "late_isi_error_ms": abs(float(np.mean(ia[-3:]) - np.mean(ib[-3:]))) if n else np.nan,
        "adaptation_ratio_error": abs(ratio(ia) - ratio(ib)),
        "observed_adaptation_ratio": ratio(ia), "predicted_adaptation_ratio": ratio(ib),
    }
