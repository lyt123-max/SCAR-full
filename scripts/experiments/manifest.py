from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.score_outputs import score_metric_groups


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_run_id(
    *,
    method: str,
    dataset: str,
    seed: int,
    stage: str,
    config: dict[str, Any],
) -> str:
    payload = {
        "method": str(method),
        "dataset": str(dataset),
        "seed": int(seed),
        "stage": str(stage),
        "config": config,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:20]


def _read_npz_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if not members:
            raise ValueError("empty npz archive")
        for member in members:
            if not member.filename.endswith(".npy") or member.file_size <= 0:
                raise ValueError(f"invalid npz member: {member.filename}")
            with archive.open(member) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(handle)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(handle)
            shapes[Path(member.filename).stem] = tuple(shape)
        if path.stat().st_size <= 64 * 1024 * 1024 and archive.testzip() is not None:
            raise ValueError("npz CRC failure")
    return shapes


@dataclass(frozen=True)
class RunSpec:
    method: str
    dataset: str
    seed: int
    stage: str
    group: str
    compute_kind: str
    config: dict[str, Any]
    command: tuple[str, ...]
    artifact_dir: Path
    required_artifacts: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return stable_run_id(
            method=self.method,
            dataset=self.dataset,
            seed=self.seed,
            stage=self.stage,
            config=self.config,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "method": self.method,
            "dataset": self.dataset,
            "seed": self.seed,
            "stage": self.stage,
            "group": self.group,
            "compute_kind": self.compute_kind,
            "config": self.config,
            "command": list(self.command),
            "artifact_dir": str(self.artifact_dir),
            "required_artifacts": list(self.required_artifacts),
            "dependencies": list(self.dependencies),
            "metadata": self.metadata,
        }


def artifact_is_complete(spec: RunSpec) -> bool:
    if not spec.required_artifacts:
        return False
    array_shapes: dict[str, tuple[int, ...]] = {}
    npz_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    json_payloads: dict[str, object] = {}
    for relative in spec.required_artifacts:
        path = spec.artifact_dir / relative
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                json_payloads[path.name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return False
        elif suffix == ".npy":
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.size <= 0:
                    return False
                array_shapes[path.name] = tuple(array.shape)
                del array
            except (OSError, ValueError, EOFError):
                return False
        elif suffix == ".npz":
            try:
                npz_shapes[path.name] = _read_npz_shapes(path)
            except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
                return False
        elif suffix == ".pt":
            try:
                header = path.read_bytes()[:4]
            except OSError:
                return False
            if not (header.startswith(b"PK") or header.startswith(b"\x80")):
                return False
    if "scores.npy" in array_shapes and "labels.npy" in array_shapes:
        if math.prod(array_shapes["scores.npy"]) != math.prod(array_shapes["labels.npy"]):
            return False
    if "scores.npy" in array_shapes:
        expected_size = math.prod(array_shapes["scores.npy"])
        for name, shape in array_shapes.items():
            if name.startswith("scores_") and math.prod(shape) != expected_size:
                return False
    diagnostic_shapes = npz_shapes.get("test_diagnostic_scores.npz", {})
    if "test_scores_selected.npy" in array_shapes and "labels" in diagnostic_shapes:
        expected_size = math.prod(array_shapes["test_scores_selected.npy"])
        if math.prod(diagnostic_shapes["labels"]) != expected_size:
            return False
        for name, shape in array_shapes.items():
            if name.startswith("test_scores_") and math.prod(shape) != expected_size:
                return False
    if "test_sequence_scores_selected.npy" in array_shapes:
        table_path = spec.artifact_dir / "test_sequence_scores.csv"
        if table_path.is_file():
            try:
                row_count = sum(
                    bool(line.strip())
                    for line in table_path.read_text(encoding="utf-8").splitlines()
                ) - 1
            except (OSError, UnicodeDecodeError):
                return False
            if row_count != math.prod(array_shapes["test_sequence_scores_selected.npy"]):
                return False
        expected_size = math.prod(array_shapes["test_sequence_scores_selected.npy"])
        for name, shape in array_shapes.items():
            if name.startswith("test_sequence_scores_") and math.prod(shape) != expected_size:
                return False
    metrics = json_payloads.get("test_metrics.json")
    has_scar_score_contract = any(
        name in array_shapes
        for name in ("test_scores_raw_max.npy", "test_sequence_scores_raw_max.npy")
    )
    if has_scar_score_contract:
        if not isinstance(metrics, dict):
            return False
        try:
            groups = score_metric_groups(metrics, require_core=True)
        except KeyError:
            return False
        expected_subscores = {
            name.removeprefix("test_scores_").removesuffix(".npy")
            for name in array_shapes
            if name.startswith("test_scores_completion_scale")
            or name in {"test_scores_knn_distance.npy", "test_scores_state_novelty.npy"}
        }
        expected_subscores.update(
            name.removeprefix("test_sequence_scores_").removesuffix(".npy")
            for name in array_shapes
            if name.startswith("test_sequence_scores_completion_scale")
            or name
            in {
                "test_sequence_scores_knn_distance.npy",
                "test_sequence_scores_state_novelty.npy",
            }
        )
        if not expected_subscores.issubset(groups):
            return False
    return True


def artifact_is_reusable(spec: RunSpec) -> bool:
    if not artifact_is_complete(spec):
        return False
    record_path = spec.artifact_dir / "run_record.json"
    if not record_path.is_file():
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        record.get("run_id") == spec.run_id
        and record.get("status") == "completed"
        and bool(record.get("artifacts_complete"))
    )
