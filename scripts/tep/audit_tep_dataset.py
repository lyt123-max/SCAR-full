#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad.data import (
    TEP_NUM_INPUT_CHANNELS,
    resolve_tep_files,
    tep_file_fault_to_idv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MMFDD-TEP files before training.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=["selected", "full"], default="full")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_name(file_name: str) -> tuple[int, int]:
    stem = Path(file_name).stem
    mode_part, file_fault_part = stem.split("d", maxsplit=1)
    return int(mode_part[1:]), int(file_fault_part)


def inspect_file(file_name: str, path: Path, seq_len: int) -> dict[str, Any]:
    payload = sio.loadmat(path)
    key = Path(file_name).stem
    if key not in payload:
        raise KeyError(f"{path} is missing expected variable {key!r}.")
    array = np.asarray(payload[key])
    if array.ndim != 2:
        raise ValueError(f"{path} must contain a 2D matrix, got {array.shape}.")
    mode_id, file_fault_id = parse_name(file_name)
    rows, columns = int(array.shape[0]), int(array.shape[1])
    process = np.asarray(array[:, : min(columns, TEP_NUM_INPUT_CHANNELS)], dtype=np.float64)
    finite = bool(np.isfinite(process).all())
    usable = bool(rows >= int(seq_len) and columns >= TEP_NUM_INPUT_CHANNELS and finite)
    return {
        "mode": mode_id,
        "file_name": file_name,
        "file_id": key,
        "kind": "normal" if file_fault_id == 0 else "fault",
        "file_fault_id": file_fault_id,
        "official_idv": tep_file_fault_to_idv(file_fault_id),
        "rows": rows,
        "columns": columns,
        "file_bytes": int(path.stat().st_size),
        "full_length_7201": bool(rows == 7201),
        "short_sequence": bool(file_fault_id > 0 and rows < 7201),
        "usable_for_seq_len": usable,
        "finite_first_53_channels": finite,
        "path": str(path.resolve()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_mode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["mode"])].append(record)
    rows: list[dict[str, Any]] = []
    for mode_id in sorted(grouped):
        group = grouped[mode_id]
        faults = [record for record in group if record["kind"] == "fault"]
        rows.append(
            {
                "mode": f"M{mode_id}",
                "normal_sequences": sum(record["kind"] == "normal" for record in group),
                "fault_sequences": len(faults),
                "full_length_faults": sum(bool(record["full_length_7201"]) for record in faults),
                "short_faults": sum(bool(record["short_sequence"]) for record in faults),
                "min_fault_length": min(int(record["rows"]) for record in faults),
                "max_fault_length": max(int(record["rows"]) for record in faults),
                "usable_faults": sum(bool(record["usable_for_seq_len"]) for record in faults),
                "available_idvs": ",".join(
                    f"IDV{value}"
                    for value in sorted(int(record["official_idv"]) for record in faults)
                ),
            }
        )
    return rows


def write_markdown(path: Path, mode_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# MMFDD-TEP Data Audit",
        "",
        f"- Protocol: `{summary['protocol']}`",
        f"- Modes: {summary['n_modes']}",
        f"- Normal sequences: {summary['n_normal_sequences']}",
        f"- Fault sequences: {summary['n_fault_sequences']}",
        f"- Full-length fault sequences: {summary['n_full_length_fault_sequences']}",
        f"- Short but usable fault sequences: {summary['n_short_fault_sequences']}",
        f"- Unusable sequences: {summary['n_unusable_sequences']}",
        "",
        "| Mode | Normal | Faults | Full length | Short | Min length | Max length | Usable | Available IDVs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in mode_rows:
        lines.append(
            "| {mode} | {normal_sequences} | {fault_sequences} | {full_length_faults} | "
            "{short_faults} | {min_fault_length} | {max_fault_length} | {usable_faults} | "
            "{available_idvs} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if int(args.seq_len) <= 0:
        raise ValueError("--seq-len must be positive.")
    file_paths, normal_files, fault_files = resolve_tep_files(args.data_root, args.protocol)
    records = [
        inspect_file(file_name, file_paths[file_name], int(args.seq_len))
        for file_name in (*normal_files, *fault_files)
    ]
    mode_rows = build_mode_rows(records)
    fault_records = [record for record in records if record["kind"] == "fault"]
    summary = {
        "schema_version": 1,
        "protocol": args.protocol,
        "data_root": str(args.data_root.resolve()),
        "seq_len": int(args.seq_len),
        "official_fault_mapping": "IDV = 29 - d for d01...d28",
        "n_modes": len(mode_rows),
        "n_files": len(records),
        "n_normal_sequences": len(normal_files),
        "n_fault_sequences": len(fault_files),
        "n_full_length_fault_sequences": sum(
            bool(record["full_length_7201"]) for record in fault_records
        ),
        "n_short_fault_sequences": sum(bool(record["short_sequence"]) for record in fault_records),
        "n_unusable_sequences": sum(not bool(record["usable_for_seq_len"]) for record in records),
        "mode_summary": mode_rows,
        "files": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tep_data_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "tep_data_audit_files.csv", records)
    write_csv(args.output_dir / "tep_data_audit_modes.csv", mode_rows)
    write_markdown(args.output_dir / "tep_data_audit.md", mode_rows, summary)
    print(
        "[TEP Audit] "
        f"protocol={args.protocol} modes={summary['n_modes']} files={summary['n_files']} "
        f"faults={summary['n_fault_sequences']} short={summary['n_short_fault_sequences']} "
        f"unusable={summary['n_unusable_sequences']} output={args.output_dir}"
    )
    return 0 if summary["n_unusable_sequences"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
