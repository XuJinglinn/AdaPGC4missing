#!/usr/bin/env python3
"""Reproducible paper t-SNE figures for AdaPGC feature records.

This module is read-only with respect to experiment directories.  It consumes
the schema written by ``utilities.exp.save_recovered_features_records`` and
``append_prediction_results`` and writes figures plus audit artifacts to a
separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PALETTE = ["#35618F", "#D49A28", "#D66B37", "#708238", "#B65A7A"]
MARKERS = ["o", "s", "^", "D", "P"]


def _require_plot_dependencies():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.metrics import balanced_accuracy_score, silhouette_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires numpy, matplotlib, scikit-learn, and the project "
            "PyTorch environment. Install the first three with "
            "`pip install numpy matplotlib scikit-learn`."
        ) from error
    return {
        "plt": plt,
        "Line2D": Line2D,
        "PCA": PCA,
        "TSNE": TSNE,
        "balanced_accuracy_score": balanced_accuracy_score,
        "silhouette_score": silhouette_score,
        "StratifiedKFold": StratifiedKFold,
        "cross_val_predict": cross_val_predict,
        "KNeighborsClassifier": KNeighborsClassifier,
    }


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is None")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(f"{name} has object dtype; expected a numeric tensor")
    return array


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Reading .pt records requires PyTorch. Run this script in the same "
            "environment used for AdaPGC experiments."
        ) from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only keyword.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a dictionary in {path}, got {type(payload)!r}")
    return payload


def _check_matrix(array: np.ndarray, name: str, rows: Optional[int] = None) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, features], got {array.shape}")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} has {array.shape[0]} rows; expected {rows}")
    if array.shape[1] < 2:
        raise ValueError(f"{name} has only {array.shape[1]} feature dimension(s)")
    if not np.isfinite(array).all():
        bad = int(np.size(array) - np.isfinite(array).sum())
        raise ValueError(f"{name} contains {bad} NaN/Inf values")
    return array


def _read_label_names(path: Optional[Path]) -> Dict[int, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Label CSV does not exist: {path}")
    names: Dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            if "index" not in row or "display_name" not in row:
                raise ValueError(
                    f"{path} must contain columns 'index' and 'display_name'"
                )
            try:
                class_id = int(row["index"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid class index on row {row_number}: {row['index']!r}") from error
            names[class_id] = str(row["display_name"]).strip()
    return names


@dataclass
class FeatureTable:
    sample_names: List[str]
    labels: np.ndarray
    features: np.ndarray
    metadata: Dict[str, Any]

    def __post_init__(self):
        self.sample_names = [str(name) for name in self.sample_names]
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.features = _check_matrix(self.features, "features", len(self.sample_names))
        if self.labels.shape != (len(self.sample_names),):
            raise ValueError(
                f"labels must have shape ({len(self.sample_names)},), got {self.labels.shape}"
            )
        duplicates = [name for name, count in Counter(self.sample_names).items() if count > 1]
        if duplicates:
            preview = ", ".join(duplicates[:5])
            raise ValueError(f"Duplicate sample names in feature table: {preview}")

    def lookup(self) -> Dict[str, int]:
        return {name: index for index, name in enumerate(self.sample_names)}

    def take_names(self, names: Sequence[str]) -> "FeatureTable":
        lookup = self.lookup()
        missing = [name for name in names if name not in lookup]
        if missing:
            raise KeyError(f"Feature table is missing {len(missing)} requested samples")
        indices = np.asarray([lookup[name] for name in names], dtype=np.int64)
        return FeatureTable(
            list(names), self.labels[indices], self.features[indices], dict(self.metadata)
        )


class ExperimentRecords:
    """Validated reader for one AdaPGC experiment directory."""

    def __init__(self, exp_dir: Path, severity: Optional[int] = None):
        self.exp_dir = Path(exp_dir).expanduser().resolve()
        self.records_dir = self.exp_dir / "recovered_features_records"
        self.index_path = self.records_dir / "index.csv"
        self.predictions_path = self.exp_dir / "predictions.csv"
        self.requested_severity = severity
        for required in (self.index_path, self.predictions_path):
            if not required.is_file():
                raise FileNotFoundError(f"Required collected-data file is missing: {required}")
        self.index_rows = self._read_index()
        self.predictions = self._read_predictions()

    def _read_index(self) -> List[Dict[str, str]]:
        with self.index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"corruption", "record_type", "source", "file"}
        if not rows:
            raise ValueError(f"Feature index is empty: {self.index_path}")
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(f"Feature index lacks columns: {sorted(missing)}")
        return rows

    def _read_predictions(self) -> Dict[Tuple[str, str], int]:
        with self.predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"corruption", "severity", "sample_name", "true_label"}
        if not rows:
            raise ValueError(f"Predictions file is empty: {self.predictions_path}")
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(f"Predictions file lacks columns: {sorted(missing)}")

        severities_by_corruption: Dict[str, set] = {}
        for row in rows:
            corruption = str(row["corruption"])
            try:
                severity = int(float(row["severity"]))
            except ValueError as error:
                raise ValueError(f"Invalid severity in {self.predictions_path}: {row['severity']!r}") from error
            severities_by_corruption.setdefault(corruption, set()).add(severity)

        chosen_severity: Dict[str, int] = {}
        for corruption, values in severities_by_corruption.items():
            if self.requested_severity is not None:
                if self.requested_severity not in values:
                    raise ValueError(
                        f"Severity {self.requested_severity} is unavailable for {corruption}; "
                        f"found {sorted(values)}"
                    )
                chosen_severity[corruption] = self.requested_severity
            else:
                chosen_severity[corruption] = max(values)
                if len(values) > 1:
                    warnings.warn(
                        f"{self.exp_dir.name}/{corruption} contains severities {sorted(values)}, "
                        f"but feature filenames do not encode severity. Assuming the last run, "
                        f"severity {max(values)}. Pass --severity explicitly and verify the run."
                    )

        labels: Dict[Tuple[str, str], int] = {}
        for row in rows:
            corruption = str(row["corruption"])
            severity = int(float(row["severity"]))
            if severity != chosen_severity[corruption]:
                continue
            key = (corruption, str(row["sample_name"]))
            label = int(float(row["true_label"]))
            if key in labels and labels[key] != label:
                raise ValueError(f"Conflicting labels for {key}: {labels[key]} versus {label}")
            labels[key] = label
        return labels

    def corruptions(self, record_type: str = "forward_results") -> List[str]:
        return sorted(
            {row["corruption"] for row in self.index_rows if row["record_type"] == record_type}
        )

    def _record_files(
        self, corruption: str, record_type: str, source: Optional[str] = None
    ) -> List[Path]:
        relative_files: List[str] = []
        for row in self.index_rows:
            if row["corruption"] != corruption or row["record_type"] != record_type:
                continue
            if source is not None and row["source"] != source:
                continue
            relative_files.append(row["file"])
        # index.csv can contain repeated rows after a resumed run; filenames are
        # authoritative and later writes overwrite the same batch file.
        unique = list(OrderedDict.fromkeys(relative_files))
        if not unique:
            suffix = f", source={source}" if source else ""
            raise ValueError(
                f"No {record_type} records for corruption={corruption}{suffix} in {self.exp_dir}"
            )
        # index.csv always stores POSIX separators; rebuild the relative path
        # from its parts so the same file works on Linux and Windows.
        paths = [self.records_dir.joinpath(*Path(item.replace("\\", "/")).parts) for item in unique]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Indexed feature file is missing: {missing[0]}")
        return paths

    def _label(self, corruption: str, sample_name: str) -> int:
        key = (corruption, sample_name)
        if key not in self.predictions:
            raise KeyError(
                f"No label for sample={sample_name!r}, corruption={corruption!r} "
                f"in {self.predictions_path}"
            )
        return self.predictions[key]

    def load_forward(
        self,
        corruption: str,
        full_only: bool = False,
        mask_kind: Optional[str] = None,
    ) -> FeatureTable:
        if full_only and mask_kind is not None:
            raise ValueError("Use either full_only=True or mask_kind, not both")
        if mask_kind is not None and mask_kind not in {
            "full", "audio_only", "video_only", "both_missing"
        }:
            raise ValueError(f"Unsupported modality mask: {mask_kind!r}")
        names: List[str] = []
        labels: List[int] = []
        arrays: List[np.ndarray] = []
        mask_kinds: List[str] = []
        source_files = self._record_files(corruption, "forward_results")
        for path in source_files:
            payload = _torch_load(path)
            batch_names = [str(item) for item in payload.get("sample_names", [])]
            features = _check_matrix(_to_numpy(payload.get("feat"), f"{path}:feat"), f"{path}:feat")
            if len(batch_names) != features.shape[0]:
                raise ValueError(f"{path}: sample_names and feat row counts differ")
            masks = payload.get("mask")
            if not isinstance(masks, Mapping):
                raise ValueError(f"{path}: mask must be a dictionary")
            bool_masks: Dict[str, np.ndarray] = {}
            for key in ("full", "audio_only", "video_only", "both_missing"):
                if key not in masks:
                    raise ValueError(f"{path}: mask is missing {key!r}")
                values = _to_numpy(masks[key], f"{path}:mask:{key}").astype(bool).reshape(-1)
                if values.shape != (len(batch_names),):
                    raise ValueError(f"{path}: mask {key!r} has wrong length")
                bool_masks[key] = values
            membership = sum(mask.astype(np.int8) for mask in bool_masks.values())
            if not np.all(membership == 1):
                raise ValueError(f"{path}: modality masks are not mutually exclusive and exhaustive")
            if mask_kind is not None:
                keep = bool_masks[mask_kind]
            elif full_only:
                keep = bool_masks["full"]
            else:
                keep = np.ones(len(batch_names), dtype=bool)
            for index in np.flatnonzero(keep):
                sample_name = batch_names[index]
                names.append(sample_name)
                labels.append(self._label(corruption, sample_name))
                mask_kinds.append(next(key for key, mask in bool_masks.items() if mask[index]))
            arrays.append(features[keep])
        return FeatureTable(
            names,
            np.asarray(labels),
            np.concatenate(arrays, axis=0),
            {
                "experiment": str(self.exp_dir),
                "corruption": corruption,
                "feature_kind": "feat",
                "selected_mask_kind": mask_kind or ("full" if full_only else "all"),
                "mask_kinds": mask_kinds,
                "source_files": [str(path) for path in source_files],
            },
        )

    def load_full_modalities(self, corruption: str) -> Dict[str, FeatureTable]:
        names: List[str] = []
        labels: List[int] = []
        collected: Dict[str, List[np.ndarray]] = {"Audio": [], "Video": [], "Fused": []}
        source_files = self._record_files(corruption, "forward_results")
        for path in source_files:
            payload = _torch_load(path)
            batch_names = [str(item) for item in payload.get("sample_names", [])]
            features = _check_matrix(_to_numpy(payload.get("feat"), f"{path}:feat"), f"{path}:feat")
            masks = payload.get("mask")
            full_mask = _to_numpy(masks["full"], f"{path}:mask:full").astype(bool).reshape(-1)
            full_names = [name for name, selected in zip(batch_names, full_mask) if selected]
            stored_full_names = [str(item) for item in payload.get("full_sample_names", full_names)]
            if stored_full_names != full_names:
                raise ValueError(f"{path}: full_sample_names does not follow the full mask order")
            ca = _check_matrix(_to_numpy(payload.get("ca"), f"{path}:ca"), f"{path}:ca", len(full_names))
            cv = _check_matrix(_to_numpy(payload.get("cv"), f"{path}:cv"), f"{path}:cv", len(full_names))
            fused = _check_matrix(features[full_mask], f"{path}:feat[full]", len(full_names))
            if not (ca.shape[1] == cv.shape[1] == fused.shape[1]):
                raise ValueError(f"{path}: ca, cv, and fused feature dimensions differ")
            names.extend(full_names)
            labels.extend(self._label(corruption, name) for name in full_names)
            collected["Audio"].append(ca)
            collected["Video"].append(cv)
            collected["Fused"].append(fused)
        return {
            key: FeatureTable(
                list(names),
                np.asarray(labels),
                np.concatenate(parts, axis=0),
                {
                    "experiment": str(self.exp_dir),
                    "corruption": corruption,
                    "feature_kind": key.lower(),
                    "source_files": [str(path) for path in source_files],
                },
            )
            for key, parts in collected.items()
        }

    def load_recovered(self, corruption: str, source: str) -> FeatureTable:
        if source not in {"a", "v"}:
            raise ValueError("Recovery source must be 'a' (audio observed) or 'v' (video observed)")
        names: List[str] = []
        labels: List[int] = []
        arrays: List[np.ndarray] = []
        warmup_skipped = 0
        source_files = self._record_files(corruption, "predict_x2f", source)
        used_files: List[str] = []
        for path in source_files:
            payload = _torch_load(path)
            batch_names = [str(item) for item in payload.get("sample_names", [])]
            warmup = bool(payload.get("warmup_fallback", False))
            alpha_value = payload.get("alpha")
            means_value = payload.get("cond_means")
            if warmup or alpha_value is None or means_value is None:
                warmup_skipped += len(batch_names)
                continue
            alpha = np.asarray(_to_numpy(alpha_value, f"{path}:alpha"), dtype=np.float64)
            means = np.asarray(_to_numpy(means_value, f"{path}:cond_means"), dtype=np.float64)
            if alpha.ndim != 2 or means.ndim != 3:
                raise ValueError(
                    f"{path}: expected alpha [B,K] and cond_means [B,K,D], "
                    f"got {alpha.shape} and {means.shape}"
                )
            if alpha.shape[:2] != means.shape[:2] or alpha.shape[0] != len(batch_names):
                raise ValueError(f"{path}: alpha, cond_means, and sample_names are misaligned")
            if not np.isfinite(alpha).all() or not np.isfinite(means).all():
                raise ValueError(f"{path}: recovery tensors contain NaN/Inf")
            row_sums = alpha.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=2e-3, rtol=2e-3):
                raise ValueError(f"{path}: alpha rows do not sum to one")
            recovered = np.einsum("bk,bkd->bd", alpha, means, optimize=True)
            recovered = _check_matrix(recovered, f"{path}:expected_recovered", len(batch_names))
            names.extend(batch_names)
            labels.extend(self._label(corruption, name) for name in batch_names)
            arrays.append(recovered)
            used_files.append(str(path))
        if not arrays:
            raise ValueError(
                f"No non-warmup conditional recovery features for {corruption}, source={source}. "
                f"Warm-up samples skipped: {warmup_skipped}."
            )
        return FeatureTable(
            names,
            np.asarray(labels),
            np.concatenate(arrays, axis=0),
            {
                "experiment": str(self.exp_dir),
                "corruption": corruption,
                "feature_kind": "expected_conditional_fused",
                "source": source,
                "warmup_samples_skipped": warmup_skipped,
                "source_files": used_files,
            },
        )


def _common_balanced_names(
    tables: Sequence[FeatureTable],
    requested_classes: Optional[Sequence[int]],
    n_classes: int,
    min_per_class: int,
    max_per_class: int,
    seed: int,
) -> Tuple[List[str], List[int], Dict[int, int]]:
    if not tables:
        raise ValueError("At least one feature table is required")
    common = set(tables[0].sample_names)
    for table in tables[1:]:
        common.intersection_update(table.sample_names)
    if not common:
        raise ValueError("The compared experiment records have no common sample_name values")

    lookups = [table.lookup() for table in tables]
    labels_by_name: Dict[str, int] = {}
    for name in sorted(common):
        labels = [int(table.labels[lookup[name]]) for table, lookup in zip(tables, lookups)]
        if len(set(labels)) != 1:
            raise ValueError(f"Label mismatch for sample {name!r}: {labels}")
        labels_by_name[name] = labels[0]
    counts = Counter(labels_by_name.values())

    if requested_classes:
        selected_classes = [int(value) for value in requested_classes]
        unavailable = [value for value in selected_classes if counts[value] < min_per_class]
        if unavailable:
            details = {value: counts[value] for value in unavailable}
            raise ValueError(
                f"Requested classes do not have at least {min_per_class} common samples: {details}"
            )
    else:
        eligible = [(label, count) for label, count in counts.items() if count >= min_per_class]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        selected_classes = [label for label, _ in eligible[:n_classes]]
        if len(selected_classes) < n_classes:
            raise ValueError(
                f"Only {len(selected_classes)} classes have >= {min_per_class} common samples; "
                f"requested {n_classes}. Available counts: {dict(counts.most_common(10))}"
            )
    if len(selected_classes) > len(PALETTE):
        raise ValueError(f"At most {len(PALETTE)} classes are allowed for a readable figure")

    rng = np.random.default_rng(seed)
    selected_names: List[str] = []
    sampled_counts: Dict[int, int] = {}
    for class_id in selected_classes:
        candidates = sorted(name for name, label in labels_by_name.items() if label == class_id)
        rng.shuffle(candidates)
        chosen = sorted(candidates[:max_per_class])
        selected_names.extend(chosen)
        sampled_counts[class_id] = len(chosen)
    selected_names.sort(key=lambda name: (selected_classes.index(labels_by_name[name]), name))
    return selected_names, selected_classes, sampled_counts


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def _joint_embedding(
    groups: Sequence[np.ndarray], seed: int, pca_dim: int, perplexity: float
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    deps = _require_plot_dependencies()
    dimensions = {group.shape[1] for group in groups}
    if len(dimensions) != 1:
        raise ValueError(f"Compared feature dimensions differ: {sorted(dimensions)}")
    counts = [group.shape[0] for group in groups]
    combined = _l2_normalize(np.concatenate(groups, axis=0))
    n_samples, n_features = combined.shape
    if n_samples < 12:
        raise ValueError(f"t-SNE needs at least 12 total points; found {n_samples}")
    effective_pca = min(int(pca_dim), n_features, n_samples - 1)
    if effective_pca < 2:
        raise ValueError("PCA preprocessing needs at least two dimensions")
    pca = deps["PCA"](n_components=effective_pca, svd_solver="auto", random_state=seed)
    reduced = pca.fit_transform(combined)
    effective_perplexity = min(float(perplexity), max(2.0, (n_samples - 1) / 3.0))
    effective_perplexity = min(effective_perplexity, n_samples - 1e-3)
    tsne_kwargs = dict(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
        method="barnes_hut",
        angle=0.5,
    )
    signature = inspect.signature(deps["TSNE"])
    if "max_iter" in signature.parameters:
        tsne_kwargs["max_iter"] = 2000
    else:
        tsne_kwargs["n_iter"] = 2000
    coordinates = deps["TSNE"](**tsne_kwargs).fit_transform(reduced)
    boundaries = np.cumsum([0] + counts)
    coord_groups = [coordinates[boundaries[i] : boundaries[i + 1]] for i in range(len(counts))]
    pca_groups = [reduced[boundaries[i] : boundaries[i + 1]] for i in range(len(counts))]
    metadata = {
        "seed": seed,
        "normalization": "row_l2",
        "pca_components": effective_pca,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "tsne_perplexity": float(effective_perplexity),
        "tsne_iterations": 2000,
        "joint_fit": True,
    }
    return coord_groups, pca_groups, metadata


def _separation_metrics(features: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, float]:
    deps = _require_plot_dependencies()
    labels = np.asarray(labels)
    counts = Counter(labels.tolist())
    if len(counts) < 2:
        raise ValueError("Separation metrics require at least two classes")
    silhouette = float(deps["silhouette_score"](features, labels, metric="euclidean"))
    min_count = min(counts.values())
    n_splits = min(5, min_count)
    if n_splits < 2:
        knn_balanced = float("nan")
    else:
        neighbors = max(1, min(7, len(labels) - math.ceil(len(labels) / n_splits)))
        cv = deps["StratifiedKFold"](n_splits=n_splits, shuffle=True, random_state=seed)
        predictions = deps["cross_val_predict"](
            deps["KNeighborsClassifier"](n_neighbors=neighbors, weights="distance"),
            features,
            labels,
            cv=cv,
        )
        knn_balanced = float(deps["balanced_accuracy_score"](labels, predictions))
    global_mean = features.mean(axis=0)
    between = 0.0
    within = 0.0
    for class_id in sorted(counts):
        class_features = features[labels == class_id]
        class_mean = class_features.mean(axis=0)
        between += len(class_features) * float(np.sum((class_mean - global_mean) ** 2))
        within += float(np.sum((class_features - class_mean) ** 2))
    fisher = between / max(within, 1e-12)
    return {
        "silhouette_pca": silhouette,
        "knn_balanced_accuracy_pca": knn_balanced,
        "fisher_ratio_pca": fisher,
    }


def _paired_representation_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    reference_norm = _l2_normalize(reference)
    candidate_norm = _l2_normalize(candidate)
    cosine = np.sum(reference_norm * candidate_norm, axis=1)
    relative_l2 = np.linalg.norm(candidate - reference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1e-12
    )
    reference_centroids = np.stack(
        [reference_norm[labels == value].mean(axis=0) for value in classes]
    )
    reference_centroids = _l2_normalize(reference_centroids)
    centroid_predictions = np.asarray(classes)[
        np.argmax(candidate_norm @ reference_centroids.T, axis=1)
    ]
    agreement = float(np.mean(centroid_predictions == labels))
    per_class: List[Dict[str, Any]] = []
    for class_id in classes:
        mask = labels == class_id
        reference_class = reference_norm[mask]
        candidate_class = candidate_norm[mask]
        reference_centroid = _l2_normalize(
            reference_class.mean(axis=0, keepdims=True)
        )[0]
        candidate_centroid = _l2_normalize(
            candidate_class.mean(axis=0, keepdims=True)
        )[0]
        within_radius = np.median(
            np.linalg.norm(reference_class - reference_centroid, axis=1)
        )
        displacement = np.linalg.norm(candidate_centroid - reference_centroid)
        per_class.append(
            {
                "class_id": int(class_id),
                "n_pairs": int(mask.sum()),
                "median_paired_cosine": float(np.median(cosine[mask])),
                "median_relative_l2": float(np.median(relative_l2[mask])),
                "centroid_cosine": float(
                    np.dot(reference_centroid, candidate_centroid)
                ),
                "centroid_displacement_over_reference_radius": float(
                    displacement / max(within_radius, 1e-12)
                ),
            }
        )
    summary = {
        "n_pairs": int(len(labels)),
        "median_paired_cosine": float(np.median(cosine)),
        "mean_paired_cosine": float(np.mean(cosine)),
        "median_relative_l2": float(np.median(relative_l2)),
        "reference_centroid_agreement": agreement,
    }
    return summary, per_class


def _configure_style(plt):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "axes.edgecolor": "#38424D",
            "axes.linewidth": 0.8,
            "text.color": "#26313C",
            "axes.labelcolor": "#26313C",
            "xtick.color": "#66717D",
            "ytick.color": "#66717D",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _class_text(class_id: int, label_names: Mapping[int, str]) -> str:
    name = label_names.get(int(class_id))
    return f"{class_id}: {name}" if name else f"Class {class_id}"


def _style_axes(ax, show_y: bool = True):
    ax.set_xlabel("t-SNE 1")
    if show_y:
        ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _save_figure(fig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return png, pdf


def plot_corruption(args) -> Dict[str, str]:
    deps = _require_plot_dependencies()
    plt = deps["plt"]
    _configure_style(plt)
    conditions: "OrderedDict[str, ExperimentRecords]" = OrderedDict()
    for specification in args.condition:
        if "=" not in specification:
            raise ValueError("Each --condition must use LABEL=EXPERIMENT_DIR")
        label, directory = specification.split("=", 1)
        label = label.strip()
        if not label or label in conditions:
            raise ValueError(f"Condition label is empty or duplicated: {label!r}")
        conditions[label] = ExperimentRecords(Path(directory), severity=args.severity)
    if len(conditions) > 3:
        raise ValueError("Use at most three method conditions in one readable figure")

    tables = OrderedDict(
        (label, records.load_forward(args.corruption)) for label, records in conditions.items()
    )
    names, classes, counts = _common_balanced_names(
        list(tables.values()), args.classes, args.n_classes, args.min_per_class,
        args.max_per_class, args.seed
    )
    selected = OrderedDict((label, table.take_names(names)) for label, table in tables.items())
    coords, pca_groups, embedding_meta = _joint_embedding(
        [table.features for table in selected.values()], args.seed, args.pca_dim, args.perplexity
    )
    metrics = []
    for (label, table), pca_features in zip(selected.items(), pca_groups):
        metrics.append({"condition": label, **_separation_metrics(pca_features, table.labels, args.seed)})

    fig, axes = plt.subplots(
        1, len(selected), figsize=(4.3 * len(selected), 4.3), sharex=True, sharey=True,
        constrained_layout=False, squeeze=False
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.76, wspace=0.04)
    axes = axes[0]
    label_names = _read_label_names(Path(args.label_csv) if args.label_csv else None)
    points_rows: List[Dict[str, Any]] = []
    legend_handles = []
    for class_index, class_id in enumerate(classes):
        legend_handles.append(
            deps["Line2D"](
                [0], [0], marker=MARKERS[class_index], linestyle="none",
                markerfacecolor=PALETTE[class_index], markeredgecolor="white",
                markeredgewidth=0.6, markersize=7,
                label=_class_text(class_id, label_names),
            )
        )
    for axis_index, ((condition, table), coordinate, metric) in enumerate(
        zip(selected.items(), coords, metrics)
    ):
        ax = axes[axis_index]
        for class_index, class_id in enumerate(classes):
            mask = table.labels == class_id
            ax.scatter(
                coordinate[mask, 0], coordinate[mask, 1], s=22,
                c=PALETTE[class_index], marker=MARKERS[class_index], alpha=0.78,
                edgecolors="white", linewidths=0.35, rasterized=True,
            )
        ax.set_title(condition, fontweight="bold", pad=8)
        _style_axes(ax, show_y=axis_index == 0)
        ax.text(
            0.02, 0.02,
            f"kNN bal. acc. {metric['knn_balanced_accuracy_pca']:.3f}\n"
            f"silhouette {metric['silhouette_pca']:.3f}",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCD3DA", alpha=0.92),
        )
        for name, class_id, xy in zip(table.sample_names, table.labels, coordinate):
            points_rows.append(
                {"condition": condition, "corruption": args.corruption,
                 "sample_name": name, "class_id": int(class_id),
                 "tsne_1": float(xy[0]), "tsne_2": float(xy[1])}
            )
    fig.suptitle(args.title or "Feature geometry under single-modality corruption",
                 fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.935,
        f"{args.corruption} | joint embedding | {len(classes)} classes | "
        f"{sum(counts.values())} matched samples per condition",
        ha="center", va="top", fontsize=8.5, color="#5F6B76",
    )
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.875),
               ncol=min(len(classes), 5), frameon=False)

    output_dir = Path(args.output_dir).expanduser().resolve()
    name = args.name or f"tsne_corruption_{args.corruption}"
    png, pdf = _save_figure(fig, output_dir, name)
    plt.close(fig)
    _write_csv(output_dir / f"{name}_points.csv", points_rows)
    _write_csv(output_dir / f"{name}_metrics.csv", metrics)
    _write_json(
        output_dir / f"{name}_manifest.json",
        {
            "figure": "corruption_discriminability",
            "corruption": args.corruption,
            "conditions": {label: str(store.exp_dir) for label, store in conditions.items()},
            "selected_classes": classes,
            "class_names": {str(value): label_names.get(value) for value in classes},
            "sample_count_by_class": counts,
            "selected_sample_names": names,
            "embedding": embedding_meta,
            "metrics_space": "shared_l2_normalized_PCA",
            "selection_rule": "largest common class support, then class id",
        },
    )
    return {"png": str(png), "pdf": str(pdf)}


def plot_recovery(args) -> Dict[str, str]:
    deps = _require_plot_dependencies()
    plt = deps["plt"]
    _configure_style(plt)
    source = args.source
    if source is None:
        if args.missing_corruption.startswith("missing_a_"):
            source = "v"
        elif args.missing_corruption.startswith("missing_v_"):
            source = "a"
        else:
            raise ValueError("Cannot infer --source; pass 'a' or 'v' explicitly")
    clean_store = ExperimentRecords(Path(args.clean_exp), severity=args.clean_severity)
    missing_store = ExperimentRecords(Path(args.missing_exp), severity=args.missing_severity)

    if source == "a":
        observed = "audio"
        missing = "video"
        available_mask = "audio_only"
    else:
        observed = "video"
        missing = "audio"
        available_mask = "video_only"

    available = missing_store.load_forward(
        args.missing_corruption, mask_kind=available_mask
    )
    recovered = missing_store.load_recovered(args.missing_corruption, source)
    clean_fused_ground_truth = clean_store.load_forward(
        args.clean_corruption, full_only=True
    )
    names, classes, counts = _common_balanced_names(
        [available, recovered, clean_fused_ground_truth],
        args.classes,
        args.n_classes,
        args.min_per_class,
        args.max_per_class, args.seed
    )
    available = available.take_names(names)
    recovered = recovered.take_names(names)
    clean_fused_ground_truth = clean_fused_ground_truth.take_names(names)
    coords, _, embedding_meta = _joint_embedding(
        [available.features, recovered.features, clean_fused_ground_truth.features],
        args.seed,
        args.pca_dim,
        args.perplexity,
    )
    available_xy, recovered_xy, clean_fused_ground_truth_xy = coords
    recovery_summary, recovery_per_class = _paired_representation_metrics(
        clean_fused_ground_truth.features, recovered.features,
        clean_fused_ground_truth.labels, classes
    )
    available_summary, available_per_class = _paired_representation_metrics(
        clean_fused_ground_truth.features, available.features,
        clean_fused_ground_truth.labels, classes
    )
    recovery_available_summary, recovery_available_per_class = (
        _paired_representation_metrics(
            available.features, recovered.features, available.labels, classes
        )
    )
    label_names = _read_label_names(Path(args.label_csv) if args.label_csv else None)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.6), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.73)
    _style_axes(ax, show_y=True)

    state_colors = {
        "available": "#35618F",
        "recovered": "#D49A28",
        "clean_fused_ground_truth": "#D66B37",
    }
    class_handles = []
    for class_index, class_id in enumerate(classes):
        mask = available.labels == class_id
        marker = MARKERS[class_index]
        class_indices = np.flatnonzero(mask)[: max(0, args.overlay_pairs_per_class)]
        for sample_index in class_indices:
            ax.plot(
                [
                    recovered_xy[sample_index, 0],
                    clean_fused_ground_truth_xy[sample_index, 0],
                ],
                [
                    recovered_xy[sample_index, 1],
                    clean_fused_ground_truth_xy[sample_index, 1],
                ],
                color="#B8C0C8", linewidth=0.55, alpha=0.34, zorder=1,
            )

        ax.scatter(
            available_xy[mask, 0], available_xy[mask, 1], s=22,
            c=state_colors["available"], marker=marker, alpha=0.55,
            edgecolors="white", linewidths=0.30, rasterized=True, zorder=2,
        )
        ax.scatter(
            clean_fused_ground_truth_xy[mask, 0],
            clean_fused_ground_truth_xy[mask, 1],
            s=27,
            c=state_colors["clean_fused_ground_truth"], marker=marker, alpha=0.66,
            edgecolors="#7D3A22", linewidths=0.35, rasterized=True, zorder=3,
        )
        ax.scatter(
            recovered_xy[mask, 0], recovered_xy[mask, 1], s=34,
            facecolors="none", edgecolors=state_colors["recovered"], marker=marker,
            linewidths=0.95, rasterized=True, zorder=4,
        )
        class_handles.append(
            deps["Line2D"](
                [0], [0], marker=marker, linestyle="none",
                markerfacecolor="#6C7680", markeredgecolor="white",
                markersize=6.5, label=_class_text(class_id, label_names),
            )
        )

    state_handles = [
        deps["Line2D"](
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=state_colors["available"], markeredgecolor="white",
            markersize=7, label=f"Available ({observed})",
        ),
        deps["Line2D"](
            [0], [0], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor=state_colors["recovered"], markeredgewidth=1.2,
            markersize=7, label="Posterior-mean recovery (fused F)",
        ),
        deps["Line2D"](
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=state_colors["clean_fused_ground_truth"],
            markeredgecolor="#7D3A22", markersize=7,
            label="Clean fused ground truth (F)",
        ),
    ]
    representation_legend = fig.legend(
        handles=state_handles, loc="upper center", bbox_to_anchor=(0.5, 0.875),
        ncol=3, frameon=False, title="Representation",
    )
    fig.add_artist(representation_legend)
    fig.legend(
        handles=class_handles, loc="upper center", bbox_to_anchor=(0.5, 0.805),
        ncol=min(len(classes), 4), frameon=False, title="Class",
    )
    ax.text(
        0.02, 0.02,
        f"Recovery ↔ clean fused GT median cosine "
        f"{recovery_summary['median_paired_cosine']:.3f}\n"
        f"Clean-fused-GT centroid agreement "
        f"{recovery_summary['reference_centroid_agreement']:.3f}",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCD3DA", alpha=0.92),
    )
    fig.suptitle(
        args.title or "Available, recovered, and clean fused representations",
                 fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.935,
        f"{args.missing_corruption} | available {observed}, missing {missing} | "
        f"recovery target clean fused F | joint embedding | {len(classes)} classes | "
        f"{len(names)} exact sample triplets",
        ha="center", va="top", fontsize=8.5, color="#5F6B76",
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    name = args.name or f"tsne_recovery_{args.missing_corruption}"
    png, pdf = _save_figure(fig, output_dir, name)
    plt.close(fig)
    point_rows: List[Dict[str, Any]] = []
    for state, role, coordinate in (
        ("available", f"observed_{observed}_only", available_xy),
        ("recovered", "posterior_mean_fused_F", recovered_xy),
        ("clean_fused_ground_truth", "clean_full_fused_F", clean_fused_ground_truth_xy),
    ):
        for sample_name, class_id, xy in zip(names, available.labels, coordinate):
            point_rows.append(
                {"state": state, "corruption": args.missing_corruption,
                 "representation_role": role,
                 "sample_name": sample_name, "class_id": int(class_id),
                 "tsne_1": float(xy[0]), "tsne_2": float(xy[1])}
            )
    _write_csv(output_dir / f"{name}_points.csv", point_rows)
    metric_rows: List[Dict[str, Any]] = []
    for comparison, summary, per_class in (
        ("recovery_to_clean_fused_ground_truth", recovery_summary, recovery_per_class),
        ("available_to_clean_fused_ground_truth", available_summary, available_per_class),
        ("recovery_to_available", recovery_available_summary, recovery_available_per_class),
    ):
        metric_rows.append({"comparison": comparison, "scope": "overall", **summary})
        metric_rows.extend(
            {"comparison": comparison, "scope": "class", **row}
            for row in per_class
        )
    _write_csv(output_dir / f"{name}_metrics.csv", metric_rows)
    _write_json(
        output_dir / f"{name}_manifest.json",
        {
            "figure": "available_recovery_clean_fused_ground_truth_joint_embedding",
            "clean_experiment": str(clean_store.exp_dir),
            "missing_experiment": str(missing_store.exp_dir),
            "clean_corruption": args.clean_corruption,
            "missing_corruption": args.missing_corruption,
            "recovery_source": source,
            "available_modality": observed,
            "available_mask": available_mask,
            "missing_modality": missing,
            "available_feature": (
                f"missing forward_results.feat filtered by mask.{available_mask}"
            ),
            "clean_fused_ground_truth_feature": (
                "clean forward_results.feat filtered by mask.full for the exact matched "
                "sample_name"
            ),
            "recovery_target_space": "fused_F",
            "recovery_summary": "posterior mean of saved class-conditional recovered features",
            "recovery_formula": "einsum('bk,bkd->bd', alpha, cond_means)",
            "classifier_note": (
                "The archived classifier marginalizes GDA scores evaluated at each conditional "
                "mean; it does not directly classify this single posterior-mean vector."
            ),
            "comparison_note": (
                "Recovered and ground-truth vectors both represent fused feature F. The "
                "available single-view feature is retained as context in the same joint "
                "embedding."
            ),
            "warmup_samples_skipped": recovered.metadata["warmup_samples_skipped"],
            "selected_classes": classes,
            "class_names": {str(value): label_names.get(value) for value in classes},
            "sample_count_by_class": counts,
            "selected_sample_names": names,
            "embedding": embedding_meta,
            "selection_rule": "largest exact-triplet class support, then class id",
            "quantitative_metrics_space": "original feature space after row normalization where noted",
        },
    )
    return {"png": str(png), "pdf": str(pdf)}


def plot_alignment(args) -> Dict[str, str]:
    deps = _require_plot_dependencies()
    plt = deps["plt"]
    _configure_style(plt)
    store = ExperimentRecords(Path(args.exp), severity=args.severity)
    tables = store.load_full_modalities(args.corruption)
    names, classes, counts = _common_balanced_names(
        list(tables.values()), args.classes, args.n_classes, args.min_per_class,
        args.max_per_class, args.seed
    )
    selected = OrderedDict((label, table.take_names(names)) for label, table in tables.items())
    coords, pca_groups, embedding_meta = _joint_embedding(
        [table.features for table in selected.values()], args.seed, args.pca_dim, args.perplexity
    )
    metrics = [
        {"representation": label, **_separation_metrics(pca, table.labels, args.seed)}
        for (label, table), pca in zip(selected.items(), pca_groups)
    ]
    label_names = _read_label_names(Path(args.label_csv) if args.label_csv else None)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), sharex=True, sharey=True,
                             constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.13, top=0.76, wspace=0.035)
    legend_handles = []
    for class_index, class_id in enumerate(classes):
        legend_handles.append(
            deps["Line2D"]([0], [0], marker=MARKERS[class_index], linestyle="none",
                            markerfacecolor=PALETTE[class_index], markeredgecolor="white",
                            markersize=7, label=_class_text(class_id, label_names))
        )
    point_rows = []
    for axis_index, ((representation, table), coordinate, metric) in enumerate(
        zip(selected.items(), coords, metrics)
    ):
        ax = axes[axis_index]
        for class_index, class_id in enumerate(classes):
            mask = table.labels == class_id
            ax.scatter(coordinate[mask, 0], coordinate[mask, 1], s=22,
                       c=PALETTE[class_index], marker=MARKERS[class_index], alpha=0.78,
                       edgecolors="white", linewidths=0.35, rasterized=True)
        ax.set_title(f"{representation} representation", fontweight="bold", pad=8)
        _style_axes(ax, show_y=axis_index == 0)
        ax.text(0.02, 0.02, f"silhouette {metric['silhouette_pca']:.3f}",
                transform=ax.transAxes, va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCD3DA", alpha=0.92))
        for sample_name, class_id, xy in zip(names, table.labels, coordinate):
            point_rows.append({"representation": representation, "sample_name": sample_name,
                               "class_id": int(class_id), "tsne_1": float(xy[0]),
                               "tsne_2": float(xy[1])})
    fig.suptitle(args.title or "Clean modality-specific and fused feature geometry",
                 fontweight="bold", y=0.985)
    fig.text(0.5, 0.935, f"{args.corruption} | joint embedding | {len(classes)} classes | "
             f"{len(names)} matched full samples per representation",
             ha="center", va="top", fontsize=8.5, color="#5F6B76")
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.875),
               ncol=min(len(classes), 4), frameon=False)
    output_dir = Path(args.output_dir).expanduser().resolve()
    name = args.name or f"tsne_alignment_{args.corruption}"
    png, pdf = _save_figure(fig, output_dir, name)
    plt.close(fig)
    _write_csv(output_dir / f"{name}_points.csv", point_rows)
    _write_csv(output_dir / f"{name}_metrics.csv", metrics)
    _write_json(output_dir / f"{name}_manifest.json", {
        "figure": "modality_alignment", "experiment": str(store.exp_dir),
        "corruption": args.corruption, "selected_classes": classes,
        "class_names": {str(value): label_names.get(value) for value in classes},
        "sample_count_by_class": counts, "selected_sample_names": names,
        "embedding": embedding_meta,
    })
    return {"png": str(png), "pdf": str(pdf)}


def inspect_experiment(args) -> Dict[str, Any]:
    store = ExperimentRecords(Path(args.exp), severity=args.severity)
    by_key: Dict[Tuple[str, str, str], int] = Counter()
    samples_by_key: Dict[Tuple[str, str, str], int] = Counter()
    warmup_by_key: Dict[Tuple[str, str, str], int] = Counter()
    for row in store.index_rows:
        key = (row["corruption"], row["record_type"], row["source"])
        by_key[key] += 1
        try:
            samples_by_key[key] += int(float(row.get("num_samples", 0) or 0))
        except ValueError:
            pass
        if str(row.get("warmup_fallback", "")).lower() == "true":
            warmup_by_key[key] += 1
    rows = [
        {"corruption": key[0], "record_type": key[1], "source": key[2],
         "index_rows": count, "indexed_samples_including_repeats": samples_by_key[key],
         "warmup_batches": warmup_by_key[key]}
        for key, count in sorted(by_key.items())
    ]
    result = {
        "experiment": str(store.exp_dir),
        "prediction_label_pairs": len(store.predictions),
        "records": rows,
        "note": "Run plotting commands for tensor-shape and exact-pair validation.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _add_selection_arguments(parser, default_classes: int):
    parser.add_argument("--classes", nargs="+", type=int, help="Explicit class IDs; disables auto-selection")
    parser.add_argument("--n-classes", type=int, default=default_classes)
    parser.add_argument("--min-per-class", type=int, default=12)
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--label-csv", help="Optional class-label CSV with index and display_name columns")
    parser.add_argument("--output-dir", default="paper_figures/tsne")
    parser.add_argument("--name", help="Output filename stem")
    parser.add_argument("--title", help="Optional neutral figure title")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create audited t-SNE figures from AdaPGC collected feature records."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Summarize a collected experiment")
    inspect_parser.add_argument("--exp", required=True)
    inspect_parser.add_argument("--severity", type=int)
    inspect_parser.set_defaults(handler=inspect_experiment)

    corruption = subparsers.add_parser(
        "corruption", help="Compare class discrimination under one corruption"
    )
    corruption.add_argument(
        "--condition", action="append", required=True,
        help="Repeat LABEL=EXPERIMENT_DIR, e.g. --condition Source=/path --condition AdaPGC=/path",
    )
    corruption.add_argument("--corruption", required=True)
    corruption.add_argument("--severity", type=int)
    _add_selection_arguments(corruption, default_classes=5)
    corruption.set_defaults(handler=plot_corruption)

    recovery = subparsers.add_parser(
        "recovery",
        help=(
            "Jointly plot available, posterior-mean recovered, and exact "
            "clean fused ground-truth representations"
        ),
    )
    recovery.add_argument("--clean-exp", required=True)
    recovery.add_argument("--missing-exp", required=True)
    recovery.add_argument("--clean-corruption", default="clean")
    recovery.add_argument("--missing-corruption", required=True)
    recovery.add_argument("--source", choices=["a", "v"],
                          help="Observed source modality; inferred from missing_a/missing_v")
    recovery.add_argument("--clean-severity", type=int)
    recovery.add_argument("--missing-severity", type=int)
    recovery.add_argument(
        "--overlay-pairs-per-class", type=int, default=6,
        help=(
            "draw this many faint recovery-to-clean-fused-GT pair lines per class; "
            "use 0 to disable"
        ),
    )
    _add_selection_arguments(recovery, default_classes=4)
    recovery.set_defaults(handler=plot_recovery)

    alignment = subparsers.add_parser(
        "alignment", help="Supplement: compare clean audio, video, and fused geometry"
    )
    alignment.add_argument("--exp", required=True)
    alignment.add_argument("--corruption", default="clean")
    alignment.add_argument("--severity", type=int)
    _add_selection_arguments(alignment, default_classes=4)
    alignment.set_defaults(handler=plot_alignment)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.command != "inspect":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
