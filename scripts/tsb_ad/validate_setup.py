from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import MANIFESTS, default_data_root, parse_file_name, read_manifest, resolve_data_dir


def _validate_csv(path: Path, edition: str) -> dict[str, int | str]:
    metadata = parse_file_name(path.name)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) < 2 or header[-1].strip().casefold() != "label":
            raise ValueError(f"{path} must end with a Label column")
        first_row = next(reader, None)
    if first_row is None or len(first_row) != len(header):
        raise ValueError(f"{path} has no valid data row")
    n_channels = len(header) - 1
    if edition == "U" and n_channels != 1:
        raise ValueError(f"{path} is in TSB-AD-U but has {n_channels} channels")
    if edition == "M" and n_channels <= 1:
        raise ValueError(f"{path} is in TSB-AD-M but has {n_channels} channels")
    return {
        "file": path.name,
        "source_dataset": str(metadata["source_dataset"]),
        "train_length": int(metadata["train_length"]),
        "n_channels": n_channels,
    }


def validate_edition(edition: str, data_root: Path, deep: bool) -> dict[str, object]:
    data_dir = resolve_data_dir(data_root, edition)
    manifests = {
        split: read_manifest(ed, split)
        for ed, split in MANIFESTS
        if ed == edition
    }
    all_names = manifests["all"]
    local_names = sorted(path.name for path in data_dir.glob("*.csv"))
    missing = sorted(set(all_names) - set(local_names))
    unexpected = sorted(set(local_names) - set(all_names))
    if missing or unexpected:
        raise ValueError(
            f"TSB-AD-{edition} local inventory differs from the official all manifest: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    tuning = set(manifests["tuning"])
    if edition == "M":
        if tuning & set(manifests["eval"]):
            raise ValueError("M tuning and eval manifests must be disjoint")
        if tuning | set(manifests["eval"]) != set(all_names):
            raise ValueError("M tuning and eval manifests must cover the all manifest")
    else:
        eval_full = set(manifests["eval_full"])
        if tuning & eval_full:
            raise ValueError("U tuning and eval_full manifests must be disjoint")
        if tuning | eval_full != set(all_names):
            raise ValueError("U tuning and eval_full manifests must cover the all manifest")
        if not set(manifests["eval"]).issubset(eval_full):
            raise ValueError("U eval manifest must be a subset of eval_full")

    checked = [_validate_csv(data_dir / name, edition) for name in (all_names if deep else all_names[:1])]
    return {
        "edition": edition,
        "data_dir": str(data_dir),
        "local_count": len(local_names),
        "manifest_counts": {split: len(names) for split, names in manifests.items()},
        "csv_files_checked": len(checked),
        "sample": checked[0],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local TSB-AD data and official split manifests.")
    parser.add_argument("--edition", choices=["M", "U", "both"], default="both")
    parser.add_argument("--m-root", type=Path, default=default_data_root("M"))
    parser.add_argument("--u-root", type=Path, default=default_data_root("U"))
    parser.add_argument("--deep", action="store_true", help="Open every CSV header instead of one sample per edition.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    editions = ("M", "U") if args.edition == "both" else (args.edition,)
    roots = {"M": args.m_root, "U": args.u_root}
    results = [validate_edition(edition, roots[edition], args.deep) for edition in editions]
    print(json.dumps({"status": "ok", "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
