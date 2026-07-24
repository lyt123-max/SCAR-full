from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ASD = tuple(f"ASD_dataset_{index}.csv" for index in range(1, 13))
REAL = ("CalIt2.csv", "CICIDS.csv", "Creditcard.csv", "GECCO.csv", "Genesis.csv", "NYC.csv")
SYNTHETIC_PREFIXES = (
    "synthetic_con",
    "synthetic_glo",
    "synthetic_sea",
    "synthetic_sha",
    "synthetic_sub_mix",
    "synthetic_tre",
)


def _time_length(path: Path) -> int:
    maximum = -1
    count = 0
    last = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "date" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no date column.")
        for row in reader:
            value = row["date"]
            try:
                maximum = max(maximum, int(float(value)))
            except ValueError:
                if value != last:
                    count += 1
                    last = value
    return maximum + 1 if maximum >= 0 else count


def audit_dataset_root(root: Path) -> dict:
    meta_path = root / "DETECT_META.csv"
    data_dir = root / "data"
    with meta_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = {row["file_name"]: row for row in csv.DictReader(handle)}
    files = {path.name: path for path in data_dir.glob("*.csv")}
    synthetic = tuple(
        sorted(
            name
            for name in files
            if any(name.startswith(prefix) for prefix in SYNTHETIC_PREFIXES)
        )
    )
    expected = (*ASD, *REAL, *synthetic)
    if len(synthetic) != 12:
        raise ValueError(f"Expected 12 synthetic files, found {len(synthetic)}.")
    missing_files = [name for name in expected if name not in files]
    missing_meta = [name for name in expected if name not in metadata]
    if missing_files or missing_meta:
        raise FileNotFoundError(
            f"CATCH extension incomplete; files={missing_files}, metadata={missing_meta}."
        )
    rows = []
    for name in expected:
        train_length = int(float(metadata[name]["train_lens"]))
        total_length = _time_length(files[name])
        if not 0 < train_length < total_length:
            raise ValueError(
                f"Invalid DETECT_META train_lens for {name}: "
                f"{train_length} not in (0, {total_length})."
            )
        rows.append(
            {
                "file_name": name,
                "train_length": train_length,
                "total_length": total_length,
            }
        )
    return {
        "schema_version": 1,
        "counts": {"asd": len(ASD), "real": len(REAL), "synthetic": len(synthetic), "total": len(rows)},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the 30 CATCH extension files.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_dataset_root(args.data_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "catch_dataset_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
