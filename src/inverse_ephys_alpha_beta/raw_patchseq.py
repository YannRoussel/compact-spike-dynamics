"""Read local Scala and Gouwens Patch-seq NWB current-clamp sweeps."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .features import FeatureConfig, extract_voltage_features
from .hh_model import Stimulus

GOUWENS_NWB_PATTERN = "ephys/ephys_files/*.nwb"
GOUWENS_MANIFEST = (
    "2020-07-08_mouse_file_manifest/2020-07-08_mouse_file_manifest.csv"
)
SCALA_NWB_PATTERN = "000008/sub-mouse-*/*.nwb"


@dataclass(frozen=True)
class CurrentClampSweep:
    """One paired response/stimulus sweep in measurement units."""

    sweep_number: int
    response_name: str
    stimulus_name: str | None
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    current_pa: np.ndarray
    sampling_rate_hz: float
    stimulus_start_ms: float | None
    stimulus_end_ms: float | None
    stimulus_amplitude_pa: float | None
    stimulus_plateau_mad_pa: float | None
    stimulus_description: str

    @property
    def stimulus_duration_ms(self) -> float:
        if self.stimulus_start_ms is None or self.stimulus_end_ms is None:
            return 0.0
        return self.stimulus_end_ms - self.stimulus_start_ms

    @property
    def has_long_square(self) -> bool:
        return (
            self.stimulus_amplitude_pa is not None
            and self.stimulus_plateau_mad_pa is not None
            and self.stimulus_duration_ms >= 100.0
            and self.stimulus_plateau_mad_pa
            <= max(1.0, 0.05 * abs(self.stimulus_amplitude_pa))
        )


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _decode(value.item())
    return str(value)


def _raw_unit(dataset: object) -> str:
    attrs = dataset.attrs
    igor_units = attrs.get("IGORWaveUnits")
    if igor_units is not None:
        values = np.asarray(igor_units).ravel()
        if len(values):
            return _decode(values[0]).strip().lower()
    return _decode(attrs.get("unit", "")).strip().lower()


def _voltage_magnitude(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return float(np.nanquantile(finite, 0.95)) if len(finite) else np.nan


def _plausible_voltage_mv(values: np.ndarray) -> np.ndarray:
    """Resolve legacy Scala sweeps stored as V, mV, or microvolt-like values."""
    magnitude = _voltage_magnitude(values)
    if np.isfinite(magnitude) and magnitude > 500.0:
        return values / 1000.0
    return values


def _as_voltage_mv(dataset: object) -> np.ndarray:
    """Convert an NWB response dataset while tolerating legacy raw-unit exports."""
    values = np.asarray(dataset[:], dtype=float)
    unit = _raw_unit(dataset)
    conversion = float(dataset.attrs.get("conversion", 1.0))
    offset = float(dataset.attrs.get("offset", 0.0))

    if unit in {"mv", "millivolt", "millivolts"}:
        return _plausible_voltage_mv(values)
    if unit in {"v", "volt", "volts"}:
        physical_volts = values * conversion + offset
        magnitude = _voltage_magnitude(physical_volts)
        # The Scala DANDI files label already-mV arrays as volts with conversion=1.
        if np.isfinite(magnitude) and magnitude <= 2.0:
            return physical_volts * 1000.0
        return _plausible_voltage_mv(physical_volts)
    return _plausible_voltage_mv(values)


def _as_current_pa(dataset: object) -> np.ndarray:
    """Convert an NWB stimulus dataset while tolerating legacy raw-unit exports."""
    values = np.asarray(dataset[:], dtype=float)
    unit = _raw_unit(dataset)
    conversion = float(dataset.attrs.get("conversion", 1.0))
    offset = float(dataset.attrs.get("offset", 0.0))

    if unit in {"pa", "picoamp", "picoamps", "picoamperes"}:
        return values
    if unit in {"a", "amp", "amps", "ampere", "amperes"}:
        physical_amperes = values * conversion + offset
        finite = physical_amperes[np.isfinite(physical_amperes)]
        maximum = float(np.nanmax(np.abs(finite))) if len(finite) else np.nan
        # The Scala DANDI files label already-pA arrays as amperes with conversion=1.
        return physical_amperes if maximum > 1.0e-3 else physical_amperes * 1.0e12
    return values


def _longest_stimulus_epoch(
    current_pa: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return the longest non-baseline current epoch in milliseconds and pA."""
    current = np.asarray(current_pa, dtype=float)
    if len(current) < 3 or not np.any(np.isfinite(current)):
        return None, None, None, None

    edge_samples = max(1, min(len(current) // 10, int(round(0.05 * sampling_rate_hz))))
    baseline_values = np.concatenate((current[:edge_samples], current[-edge_samples:]))
    baseline = float(np.nanmedian(baseline_values))
    delta = current - baseline
    peak = float(np.nanmax(np.abs(delta)))
    if not np.isfinite(peak) or peak < 0.5:
        return None, None, None, None

    active = np.abs(delta) >= max(0.5, 0.05 * peak)
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    if not len(starts):
        return None, None, None, None
    durations = stops - starts
    index = int(np.argmax(durations))
    start, stop = int(starts[index]), int(stops[index])
    amplitude = float(np.nanmedian(current[start:stop]) - baseline)
    plateau_mad = float(
        np.nanmedian(np.abs(current[start:stop] - np.nanmedian(current[start:stop])))
    )
    return (
        1000.0 * start / sampling_rate_hz,
        1000.0 * stop / sampling_rate_hz,
        amplitude,
        plateau_mad,
    )


def _groups_by_sweep_number(group: object) -> dict[int, tuple[str, object]]:
    result: dict[int, tuple[str, object]] = {}
    for name, series in group.items():
        if "data" not in series:
            continue
        sweep_number = int(series.attrs.get("sweep_number", -1))
        if sweep_number >= 0:
            result[sweep_number] = (name, series)
    return result


def _is_long_square(
    start_ms: float | None,
    end_ms: float | None,
    amplitude_pa: float | None,
    plateau_mad_pa: float | None,
) -> bool:
    if (
        start_ms is None
        or end_ms is None
        or amplitude_pa is None
        or plateau_mad_pa is None
    ):
        return False
    return (
        end_ms - start_ms >= 100.0
        and plateau_mad_pa <= max(1.0, 0.05 * abs(amplitude_pa))
    )


def read_current_clamp_sweeps(
    path: str | Path,
    long_square_only: bool = False,
) -> list[CurrentClampSweep]:
    """Read all current-clamp sweeps from an NWB HDF5 file.

    ``h5py`` is imported lazily so processed-feature workflows do not require it.
    """
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "Reading local NWB files requires h5py; install the 'nwb' extra."
        ) from error

    sweeps: list[CurrentClampSweep] = []
    with h5py.File(Path(path), "r") as handle:
        responses = _groups_by_sweep_number(handle["acquisition"])
        presentations = _groups_by_sweep_number(handle["stimulus/presentation"])
        for sweep_number, (response_name, response) in sorted(responses.items()):
            neurodata_type = _decode(response.attrs.get("neurodata_type", ""))
            if neurodata_type not in {"CurrentClampSeries", "IZeroClampSeries"}:
                continue
            rate = float(response["starting_time"].attrs["rate"])

            stimulus_match = presentations.get(sweep_number)
            if stimulus_match is None:
                if long_square_only:
                    continue
                stimulus_name = None
                current_pa = np.zeros(len(response["data"]), dtype=float)
                start_ms = end_ms = amplitude_pa = plateau_mad_pa = None
            else:
                stimulus_name, stimulus = stimulus_match
                current_pa = _as_current_pa(stimulus["data"])
                start_ms, end_ms, amplitude_pa, plateau_mad_pa = _longest_stimulus_epoch(
                    current_pa,
                    rate,
                )
                if long_square_only and not _is_long_square(
                    start_ms,
                    end_ms,
                    amplitude_pa,
                    plateau_mad_pa,
                ):
                    continue

            voltage_mv = _as_voltage_mv(response["data"])
            n_samples = min(len(voltage_mv), len(current_pa))
            voltage_mv = voltage_mv[:n_samples]
            current_pa = current_pa[:n_samples]
            time_ms = np.arange(n_samples, dtype=float) * 1000.0 / rate

            sweeps.append(
                CurrentClampSweep(
                    sweep_number=sweep_number,
                    response_name=response_name,
                    stimulus_name=stimulus_name,
                    time_ms=time_ms,
                    voltage_mv=voltage_mv,
                    current_pa=current_pa,
                    sampling_rate_hz=rate,
                    stimulus_start_ms=start_ms,
                    stimulus_end_ms=end_ms,
                    stimulus_amplitude_pa=amplitude_pa,
                    stimulus_plateau_mad_pa=plateau_mad_pa,
                    stimulus_description=_decode(
                        response.attrs.get("stimulus_description", "")
                    ),
                )
            )
    return sweeps


def _spike_count(sweep: CurrentClampSweep, config: FeatureConfig) -> int:
    if (
        sweep.stimulus_start_ms is None
        or sweep.stimulus_end_ms is None
        or sweep.stimulus_amplitude_pa is None
    ):
        return 0
    stimulus_mask = (
        (sweep.time_ms >= sweep.stimulus_start_ms)
        & (sweep.time_ms < sweep.stimulus_end_ms)
    )
    candidate = sweep.voltage_mv[stimulus_mask]
    if len(candidate) < 3:
        return 0
    from scipy.signal import find_peaks

    dt_ms = 1000.0 / sweep.sampling_rate_hz
    peaks, _ = find_peaks(
        candidate,
        height=config.min_peak_voltage_mv,
        prominence=config.min_prominence_mv,
        distance=max(1, int(round(config.min_spike_distance_ms / dt_ms))),
    )
    return int(len(peaks))


def count_sweep_spikes(
    sweep: CurrentClampSweep,
    config: FeatureConfig | None = None,
) -> int:
    """Count spikes in one current-clamp sweep using project-wide settings."""
    return _spike_count(
        sweep,
        config or FeatureConfig(min_spikes=1),
    )


def select_rheobase_sweep(
    sweeps: Iterable[CurrentClampSweep],
    config: FeatureConfig | None = None,
) -> tuple[CurrentClampSweep | None, int]:
    """Select the lowest positive long-square step with at least one spike."""
    config = config or FeatureConfig(min_spikes=1)
    candidates = sorted(
        (
            sweep
            for sweep in sweeps
            if sweep.has_long_square
            and sweep.stimulus_amplitude_pa is not None
            and sweep.stimulus_amplitude_pa > 0.0
        ),
        key=lambda sweep: (float(sweep.stimulus_amplitude_pa), sweep.sweep_number),
    )
    for sweep in candidates:
        count = _spike_count(sweep, config)
        if count:
            return sweep, count
    return None, 0


def infer_scala_cell_id(path: str | Path) -> str:
    """Map a Scala NWB filename to the release's ``YYYYMMDD_sample_N`` ID."""
    match = re.search(r"_ses-(\d{8})-sample-(\d+)_", Path(path).name)
    if not match:
        return Path(path).stem
    return f"{match.group(1)}_sample_{int(match.group(2))}"


def load_gouwens_file_id_map(manifest_path: str | Path) -> dict[str, str]:
    """Map Gouwens NWB basenames to Allen cell specimen IDs."""
    manifest = pd.read_csv(manifest_path, dtype={"cell_specimen_id": "string"})
    rows = manifest.loc[manifest["file_type"].eq("nwb"), ["file_name", "cell_specimen_id"]]
    return dict(zip(rows["file_name"].astype(str), rows["cell_specimen_id"].astype(str)))


def discover_local_nwb(
    scala_root: str | Path | None = None,
    gouwens_root: str | Path | None = None,
) -> pd.DataFrame:
    """Inventory local raw files and resolve their biological cell identifiers."""
    records: list[dict[str, object]] = []
    if scala_root is not None:
        root = Path(scala_root)
        for path in sorted(root.glob(SCALA_NWB_PATTERN)):
            records.append(
                {
                    "dataset": "scala_room_temperature",
                    "cell_id": infer_scala_cell_id(path),
                    "nwb_path": str(path),
                }
            )

    if gouwens_root is not None:
        root = Path(gouwens_root)
        manifest_path = root / GOUWENS_MANIFEST
        id_map = (
            load_gouwens_file_id_map(manifest_path)
            if manifest_path.exists()
            else {}
        )
        for path in sorted(root.glob(GOUWENS_NWB_PATTERN)):
            records.append(
                {
                    "dataset": "gouwens_visp",
                    "cell_id": id_map.get(path.name, pd.NA),
                    "nwb_path": str(path),
                }
            )
    return pd.DataFrame.from_records(records)


def extract_rheobase_spike_cycle(
    path: str | Path,
    dataset: str,
    cell_id: str,
    config: FeatureConfig | None = None,
) -> Mapping[str, object]:
    """Extract measurement-matched spike-cycle features at sampled rheobase."""
    config = config or FeatureConfig(min_spikes=1)
    sweep, spike_count = select_rheobase_sweep(
        read_current_clamp_sweeps(path, long_square_only=True),
        config=config,
    )
    record: dict[str, object] = {
        "dataset": dataset,
        "cell_id": cell_id,
        "nwb_path": str(path),
        "raw_status": "no_spiking_long_square",
    }
    if sweep is None:
        return record

    stimulus = Stimulus(
        amplitude_pa=sweep.stimulus_amplitude_pa,
        start_ms=float(sweep.stimulus_start_ms),
        end_ms=float(sweep.stimulus_end_ms),
    )
    features = extract_voltage_features(
        sweep.time_ms,
        sweep.voltage_mv,
        stimulus,
        config=config,
    )
    record.update(
        {
            "raw_status": "ok",
            "sweep_number": sweep.sweep_number,
            "sampling_rate_hz": sweep.sampling_rate_hz,
            "stimulus_start_ms": sweep.stimulus_start_ms,
            "stimulus_end_ms": sweep.stimulus_end_ms,
            "sampled_rheobase_pa": sweep.stimulus_amplitude_pa,
            "stimulus_plateau_mad_pa": sweep.stimulus_plateau_mad_pa,
            "spike_count": spike_count,
        }
    )
    record.update({f"feature__{name}": value for name, value in features.items()})
    return record


def extract_local_spike_cycles(
    inventory: pd.DataFrame,
    limit_per_dataset: int | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Extract sampled-rheobase spike-cycle features for an NWB inventory."""
    required = {"dataset", "cell_id", "nwb_path"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"Inventory is missing columns: {sorted(missing)}")
    if limit_per_dataset is not None and limit_per_dataset < 1:
        raise ValueError("limit_per_dataset must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    selected = inventory.copy()
    if limit_per_dataset is not None:
        selected = selected.groupby("dataset", sort=False, dropna=False).head(
            limit_per_dataset
        )

    tasks = [
        (str(row.dataset), str(row.cell_id), str(row.nwb_path))
        for row in selected.itertuples(index=False)
    ]
    if workers == 1:
        records = [_extract_inventory_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_extract_inventory_task, tasks))
    return pd.DataFrame.from_records(records)


def _extract_inventory_task(
    task: tuple[str, str, str],
) -> Mapping[str, object]:
    dataset, cell_id, nwb_path = task
    try:
        return extract_rheobase_spike_cycle(
            nwb_path,
            dataset=dataset,
            cell_id=cell_id,
        )
    except Exception as error:
        return {
            "dataset": dataset,
            "cell_id": cell_id,
            "nwb_path": nwb_path,
            "raw_status": f"error:{type(error).__name__}",
            "raw_error": str(error),
        }
