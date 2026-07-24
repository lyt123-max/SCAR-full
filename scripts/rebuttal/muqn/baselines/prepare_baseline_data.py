from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one CoReM-AD dataset in PaAno/DAMP-compatible form."
    )
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def export_baseline_arrays(
    *,
    train: np.ndarray,
    test: np.ndarray,
    test_labels: np.ndarray,
    dataset: str,
    output_dir: Path,
) -> dict[str, object]:
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    labels_test = (np.asarray(test_labels).reshape(-1) > 0).astype(np.int8)
    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("Baseline train/test arrays must be two-dimensional.")
    if train.shape[1] != test.shape[1]:
        raise ValueError("Baseline train/test channel counts do not match.")
    if len(test) != len(labels_test):
        raise ValueError("Test values and labels do not have the same length.")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Baseline input contains NaN or infinite values.")

    output_dir.mkdir(parents=True, exist_ok=True)
    common_dir = output_dir / "common"
    paano_dir = output_dir / "paano"
    damp_dir = output_dir / "damp"
    memto_dir = output_dir / "memto"
    for directory in (common_dir, paano_dir, damp_dir, memto_dir):
        directory.mkdir(parents=True, exist_ok=True)

    np.save(common_dir / "train.npy", train)
    np.save(common_dir / "test.npy", test)
    np.save(common_dir / "labels.npy", labels_test)

    labels_full = np.concatenate(
        [np.zeros(len(train), dtype=np.int8), labels_test]
    )
    values_full = np.concatenate([train, test], axis=0)
    combined = np.concatenate([values_full, labels_full[:, None]], axis=1)
    safe_dataset = "".join(char if char.isalnum() else "-" for char in dataset)
    filename = f"001_{safe_dataset}_id_0_Sensor_tr_{len(train)}_1st_0.csv"
    header = ",".join([*(str(index) for index in range(values_full.shape[1])), "Label"])
    np.savetxt(
        paano_dir / filename,
        combined,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.9g",
    )
    np.savetxt(
        damp_dir / f"{safe_dataset}.csv",
        combined,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.9g",
    )

    upper_dataset = dataset.upper()
    if upper_dataset in {"MSL", "SMAP", "SMD"}:
        np.save(memto_dir / f"{upper_dataset}_train.npy", train)
        np.save(memto_dir / f"{upper_dataset}_test.npy", test)
        np.save(memto_dir / f"{upper_dataset}_test_label.npy", labels_test)
    elif upper_dataset in {"PSM", "SWAT"}:
        index_train = np.arange(len(train), dtype=np.int64)[:, None]
        index_test = np.arange(len(test), dtype=np.int64)[:, None]
        train_table = np.concatenate([index_train, train], axis=1)
        test_table = np.concatenate([index_test, test], axis=1)
        label_table = np.column_stack([np.arange(len(labels_test)), labels_test])
        train_header = ",".join(["timestamp", *(f"channel_{i}" for i in range(train.shape[1]))])
        test_header = train_header
        np.savetxt(
            memto_dir / "train.csv",
            train_table,
            delimiter=",",
            header=train_header,
            comments="",
            fmt="%.9g",
        )
        np.savetxt(
            memto_dir / "test.csv",
            test_table,
            delimiter=",",
            header=test_header,
            comments="",
            fmt="%.9g",
        )
        np.savetxt(
            memto_dir / "test_label.csv",
            label_table,
            delimiter=",",
            header="timestamp,label",
            comments="",
            fmt="%d",
        )
    else:
        raise ValueError(f"Unsupported formal baseline dataset: {dataset}.")

    return {
        "schema_version": 2,
        "dataset": dataset,
        "train_end": int(len(train)),
        "test_length": int(len(test)),
        "n_channels": int(train.shape[1]),
        "label_column": int(train.shape[1]),
        "common_dir": str(common_dir),
        "paano_csv": str(paano_dir / filename),
        "damp_csv": str(damp_dir / f"{safe_dataset}.csv"),
        "memto_dir": str(memto_dir),
        "normalization": "delegated_to_official_baseline",
        "array_hashes": {
            "train": _sha256_array(train),
            "test": _sha256_array(test),
            "labels": _sha256_array(labels_test),
        },
    }


def main() -> None:
    from coremad import CoReMADConfig, CoReMADTrainer

    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    config = CoReMADConfig.load(experiment_dir / "config.json")
    config.artifact_root = str(experiment_dir.parent)
    config.experiment_name = experiment_dir.name
    trainer = CoReMADTrainer(config)
    raw = trainer._load_raw_bundle()
    train = np.asarray(raw.train, dtype=np.float32)
    test = np.asarray(raw.test, dtype=np.float32)
    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    output_dir = args.output_dir.resolve()
    metadata = export_baseline_arrays(
        train=train,
        test=test,
        test_labels=np.asarray(raw.test_labels),
        dataset=config.dataset,
        output_dir=output_dir,
    )
    (output_dir / "baseline_data_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[muQn] wrote {output_dir / 'baseline_data_manifest.json'}")


if __name__ == "__main__":
    main()
