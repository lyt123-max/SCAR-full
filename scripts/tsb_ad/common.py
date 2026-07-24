from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_SEED = 42
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"

MANIFESTS = {
    ("M", "all"): ("TSB-AD-M.csv", 200),
    ("M", "tuning"): ("TSB-AD-M-Tuning.csv", 20),
    ("M", "eval"): ("TSB-AD-M-Eva.csv", 180),
    ("U", "all"): ("TSB-AD-U.csv", 870),
    ("U", "tuning"): ("TSB-AD-U-Tuning.csv", 48),
    ("U", "eval"): ("TSB-AD-U-Eva.csv", 350),
    ("U", "eval_full"): ("TSB-AD-U-Eva-Full.csv", 822),
}

TSB_AD_FILE_PATTERN = re.compile(
    r"^(?P<index>\d{3})_(?P<source>.+?)_id_(?P<series_id>\d+)_"
    r"(?P<domain>.+?)_tr_(?P<train_length>\d+)_1st_(?P<first_anomaly>\d+)\.csv$",
    flags=re.IGNORECASE,
)


def normalize_edition(value: str) -> str:
    edition = str(value).strip().upper()
    if edition not in {"M", "U"}:
        raise ValueError(f"edition must be M or U, got {value!r}")
    return edition


def normalize_split(value: str) -> str:
    split = str(value).strip().lower().replace("-", "_")
    if split not in {"all", "tuning", "eval", "eval_full"}:
        raise ValueError(f"unsupported split {value!r}")
    return split


def manifest_spec(edition: str, split: str) -> tuple[Path, int]:
    key = (normalize_edition(edition), normalize_split(split))
    if key not in MANIFESTS:
        valid = ", ".join(split_name for ed, split_name in MANIFESTS if ed == key[0])
        raise ValueError(f"split {split!r} is not available for TSB-AD-{key[0]}; choose {valid}")
    file_name, count = MANIFESTS[key]
    return MANIFEST_DIR / file_name, count


def parse_file_name(file_name: str) -> dict[str, str | int]:
    name = Path(str(file_name)).name
    match = TSB_AD_FILE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid official TSB-AD file name: {file_name!r}")
    fields = match.groupdict()
    return {
        "file": name,
        "file_stem": Path(name).stem,
        "file_index": int(fields["index"]),
        "source_dataset": fields["source"],
        "series_id": int(fields["series_id"]),
        "domain": fields["domain"],
        "train_length": int(fields["train_length"]),
        "first_anomaly_index": int(fields["first_anomaly"]),
    }


def read_manifest(edition: str, split: str) -> list[str]:
    path, expected_count = manifest_spec(edition, split)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["file_name"]:
            raise ValueError(f"{path} must contain exactly one 'file_name' column")
        names = [str(row["file_name"]).strip() for row in reader if str(row["file_name"]).strip()]
    if len(names) != expected_count:
        raise ValueError(f"{path} contains {len(names)} rows; expected {expected_count}")
    if len(set(names)) != len(names):
        raise ValueError(f"{path} contains duplicate file names")
    for name in names:
        parse_file_name(name)
    return names


def default_data_root(edition: str) -> Path:
    edition = normalize_edition(edition)
    return PROJECT_ROOT / "dataset" / f"TSB-AD-{edition}"


def resolve_data_dir(root: str | Path, edition: str) -> Path:
    edition = normalize_edition(edition)
    base = Path(root).expanduser().resolve()
    candidates = (
        base,
        base / f"TSB-AD-{edition}",
        base / f"TSB-AD-{edition}" / f"TSB-AD-{edition}",
        base / "dataset" / f"TSB-AD-{edition}",
        base / "dataset" / f"TSB-AD-{edition}" / f"TSB-AD-{edition}",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(TSB_AD_FILE_PATTERN.fullmatch(path.name) for path in candidate.glob("*.csv")):
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot find TSB-AD-{edition} CSV directory under {base}; "
        f"expected a directory containing official numbered CSV files"
    )


def validate_local_files(data_dir: Path, expected_names: Iterable[str]) -> None:
    missing = [name for name in expected_names if not (data_dir / name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} manifest files are missing under {data_dir}: {preview}")
