"""Local Patch-seq expression loaders with explicit cell-ID provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat


@dataclass(frozen=True)
class PatchSeqExpression:
    dataset: str
    cell_ids: np.ndarray
    genes: np.ndarray
    values: sparse.csr_matrix | np.ndarray
    donor_ids: np.ndarray
    transcriptomic_types: np.ndarray
    already_log_normalized: bool
    library_size: np.ndarray | None = None
    source_cell_count: int | None = None


def is_scn_gene(gene: object) -> bool:
    """Return whether a symbol denotes a voltage-gated sodium family gene."""
    return re.match(r"^scn\d", str(gene).lower()) is not None


def is_kcn_gene(gene: object) -> bool:
    """Return whether a symbol belongs to a potassium-channel Kcn family."""
    return str(gene).lower().startswith("kcn")


def is_target_channel_gene(gene: object) -> bool:
    return is_scn_gene(gene) or is_kcn_gene(gene)


def broad_transcriptomic_class(label: object) -> str:
    """Map dataset-specific t-types to reproducible broad subclasses."""
    text = str(label).strip()
    for inhibitory in ("Pvalb", "Sst", "Vip", "Lamp5", "Sncg"):
        if text.startswith(inhibitory):
            return inhibitory
    if text.startswith("L6b"):
        return "Glut_L6b"
    if re.search(r"\bIT(?:_|$)", text):
        return "Glut_IT"
    if re.search(r"\b(?:PT|ET)(?:_|$)", text):
        return "Glut_ET"
    if re.search(r"\bCT(?:_| |$)", text):
        return "Glut_CT"
    if re.search(r"\bNP(?:_| |$)", text):
        return "Glut_NP"
    return "Other"


def _metadata_lookup(
    metadata: pd.DataFrame,
    key_column: str,
    cell_ids: np.ndarray,
    value_column: str,
) -> np.ndarray:
    lookup = (
        metadata.drop_duplicates(key_column)
        .set_index(key_column)[value_column]
    )
    return np.asarray(
        [
            lookup.get(cell_id, pd.NA)
            for cell_id in cell_ids
        ],
        dtype=object,
    )


def load_gouwens_expression(
    mat_path: str | Path,
    metadata_path: str | Path,
    channel_cpm_path: str | Path | None = None,
    requested_cell_ids: set[str] | None = None,
    channel_cache_path: str | Path | None = None,
) -> PatchSeqExpression:
    """Load preprocessed Gouwens expression with true specimen IDs."""
    data = loadmat(mat_path, squeeze_me=True)
    expression = np.asarray(data["T_dat"], dtype=float)
    ephys = np.column_stack(
        (data["E_pc_scaled"], data["E_feature"])
    )
    finite_ephys = np.isfinite(ephys).all(axis=1)
    cell_ids = np.asarray(
        data["T_spec_id_label"][finite_ephys],
        dtype=np.int64,
    ).astype(str)
    source_cell_count = len(cell_ids)
    expression = expression[finite_ephys]
    transcriptomic_types = np.asarray(
        data["cluster"][finite_ephys],
        dtype=object,
    )
    if requested_cell_ids is not None:
        requested = np.asarray(
            [cell_id in requested_cell_ids for cell_id in cell_ids],
            dtype=bool,
        )
        cell_ids = cell_ids[requested]
        expression = expression[requested]
        transcriptomic_types = transcriptomic_types[requested]
    metadata = pd.read_csv(
        metadata_path,
        dtype={
            "cell_specimen_id": "string",
            "donor_id": "string",
        },
    )
    donor_ids = _metadata_lookup(
        metadata,
        "cell_specimen_id",
        cell_ids,
        "donor_id",
    )
    genes = np.asarray(data["gene_id"], dtype=str)
    if channel_cpm_path is not None:
        channel_genes, channel_values = _gouwens_channel_expression(
            channel_cpm_path,
            metadata,
            cell_ids,
            channel_cache_path,
        )
        novel = ~np.isin(channel_genes, genes)
        if np.any(novel):
            genes = np.concatenate((genes, channel_genes[novel]))
            expression = np.column_stack(
                (expression, channel_values[:, novel])
            )
    return PatchSeqExpression(
        dataset="gouwens_visp",
        cell_ids=cell_ids,
        genes=genes,
        values=expression,
        donor_ids=donor_ids,
        transcriptomic_types=transcriptomic_types,
        already_log_normalized=True,
        source_cell_count=source_cell_count,
    )


def _gouwens_channel_expression(
    cpm_path: str | Path,
    metadata: pd.DataFrame,
    cell_ids: np.ndarray,
    cache_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        stored = np.load(cache, allow_pickle=False)
        cached_ids = stored["cell_ids"].astype(str)
        lookup = {
            cell_id: index
            for index, cell_id in enumerate(cached_ids)
        }
        if all(cell_id in lookup for cell_id in cell_ids):
            indices = np.asarray(
                [lookup[cell_id] for cell_id in cell_ids],
                dtype=int,
            )
            return (
                stored["genes"].astype(str),
                stored["values"][indices],
            )

    sample_lookup = (
        metadata.drop_duplicates("cell_specimen_id")
        .set_index("cell_specimen_id")["transcriptomics_sample_id"]
    )
    sample_ids = np.asarray(
        [sample_lookup.get(cell_id, pd.NA) for cell_id in cell_ids],
        dtype=object,
    )
    if pd.isna(sample_ids).any():
        missing = cell_ids[pd.isna(sample_ids)]
        raise ValueError(
            "Missing transcriptomic sample IDs for "
            + ", ".join(missing[:5])
        )
    sample_ids = sample_ids.astype(str)
    sample_set = set(sample_ids)

    selected_chunks = []
    reader = pd.read_csv(
        cpm_path,
        index_col=0,
        usecols=lambda column: (
            str(column).startswith("Unnamed:")
            or str(column) in sample_set
        ),
        chunksize=2000,
    )
    for chunk in reader:
        names = chunk.index.astype(str)
        keep = np.asarray(
            [is_target_channel_gene(name) for name in names],
            dtype=bool,
        )
        if np.any(keep):
            selected_chunks.append(chunk.loc[keep])
    if not selected_chunks:
        raise ValueError("No Scn/Kcn rows found in Gouwens CPM table")
    channels = pd.concat(selected_chunks).groupby(level=0).mean()
    channels = channels.loc[:, sample_ids]
    raw = channels.to_numpy(dtype=float).T
    minimum_detection = max(5, int(np.ceil(0.01 * len(cell_ids))))
    keep = (
        np.sum(raw > 0.0, axis=0) >= minimum_detection
    ) & (np.std(np.log2(raw + 1.0), axis=0) > 1e-8)
    genes = channels.index.to_numpy(dtype=str)[keep]
    values = np.log2(raw[:, keep] + 1.0)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            cell_ids=cell_ids.astype(str),
            genes=genes,
            values=values,
        )
    return genes, values


def load_scala_expression(
    pickle_path: str | Path,
    metadata_path: str | Path,
    include_channel_genes: bool = True,
) -> PatchSeqExpression:
    """Load Scala read counts, retaining its published 1000-gene mask."""
    with Path(pickle_path).open("rb") as stream:
        data = pickle.load(stream)
    counts = sparse.csr_matrix(data["counts"])
    selected = np.asarray(data["mostVariableGenes"], dtype=bool)
    genes = np.asarray(data["genes"], dtype=str)
    if include_channel_genes:
        channel_family = np.asarray(
            [is_target_channel_gene(gene) for gene in genes],
            dtype=bool,
        )
        minimum_detection = max(
            5,
            int(np.ceil(0.01 * counts.shape[0])),
        )
        detected = np.asarray(counts.getnnz(axis=0)).reshape(-1)
        selected |= channel_family & (detected >= minimum_detection)
    library_size = np.asarray(counts.sum(axis=1)).reshape(-1)
    cell_ids = np.asarray(data["cells"], dtype=str)
    metadata = pd.read_csv(
        metadata_path,
        dtype={"Cell": "string", "Mouse": "string"},
    )
    donor_ids = _metadata_lookup(
        metadata,
        "Cell",
        cell_ids,
        "Mouse",
    )
    return PatchSeqExpression(
        dataset="scala_room_temperature",
        cell_ids=cell_ids,
        genes=genes[selected],
        values=counts[:, selected],
        donor_ids=donor_ids,
        transcriptomic_types=np.asarray(data["ttype"], dtype=object),
        already_log_normalized=False,
        library_size=library_size,
        source_cell_count=len(cell_ids),
    )


def expression_matrix(data: PatchSeqExpression) -> np.ndarray:
    """Return the paper-compatible dense expression representation."""
    values = (
        data.values.toarray()
        if sparse.issparse(data.values)
        else np.asarray(data.values, dtype=float)
    )
    if data.already_log_normalized:
        return np.asarray(values, dtype=float)
    if data.library_size is None:
        raise ValueError("Raw counts require library sizes")
    scale = 1e6 / np.maximum(data.library_size, 1.0)
    return np.log2(values * scale[:, None] + 1.0)


def align_expression_to_parameters(
    expression: PatchSeqExpression,
    parameters: pd.DataFrame,
) -> tuple[PatchSeqExpression, pd.DataFrame]:
    """Align expression rows to fitted models in model-table order."""
    subset = parameters.loc[
        parameters["dataset"].eq(expression.dataset)
    ].copy()
    subset["cell_id"] = subset["cell_id"].astype(str)
    expression_index = {
        str(cell_id): index
        for index, cell_id in enumerate(expression.cell_ids)
    }
    subset = subset.loc[
        subset["cell_id"].isin(expression_index)
    ].reset_index(drop=True)
    indices = np.asarray(
        [expression_index[cell_id] for cell_id in subset["cell_id"]],
        dtype=int,
    )
    values = expression.values[indices]
    library_size = (
        expression.library_size[indices]
        if expression.library_size is not None
        else None
    )
    aligned = PatchSeqExpression(
        dataset=expression.dataset,
        cell_ids=expression.cell_ids[indices],
        genes=expression.genes,
        values=values,
        donor_ids=expression.donor_ids[indices],
        transcriptomic_types=expression.transcriptomic_types[indices],
        already_log_normalized=expression.already_log_normalized,
        library_size=library_size,
        source_cell_count=expression.source_cell_count,
    )
    return aligned, subset
