#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import requests

# The GitHub repo stores M1...M6 directly under the repo root.
DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "Lichen0102/Multi-mode-Fault-Diagnosis-Datasets-with-TE-process/main/"
)
DEFAULT_SAVE_FOLDER = "./TEP-DATA/TEP_Selected_Data"
DEFAULT_MODES = [1, 3, 4]
DEFAULT_FAULTS = [0, 1, 4, 6, 7, 10, 13, 14, 27]
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
CHUNK_SIZE = 1024 * 1024


def official_idv_for_file_fault(file_fault_id: int) -> int:
    file_fault_id = int(file_fault_id)
    if file_fault_id == 0:
        return 0
    if not 1 <= file_fault_id <= 28:
        raise ValueError(f"TEP file fault id must be in [0, 28], got {file_fault_id}.")
    return 29 - file_fault_id


def format_target(mode: int, file_fault_id: int) -> str:
    if file_fault_id == 0:
        return f"M{mode}-d00 (normal)"
    return f"M{mode}-d{file_fault_id:02d} (official IDV{official_idv_for_file_fault(file_fault_id)})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected TEP multi-mode .mat files.")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Raw file base URL of the TEP dataset repo.")
    parser.add_argument("--save-folder", type=str, default=DEFAULT_SAVE_FOLDER, help="Local folder to save downloaded .mat files.")
    parser.add_argument("--modes", type=int, nargs="+", default=DEFAULT_MODES, help="Mode ids to download, e.g. --modes 1 3 4")
    parser.add_argument(
        "--faults",
        type=int,
        nargs="+",
        default=DEFAULT_FAULTS,
        help="Upstream dXX file numbers, not physical IDV numbers; e.g. --faults 0 1 4 6 7 10 13 14 27",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry count for each file.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist locally.")
    return parser.parse_args()


def build_file_name(mode: int, fault: int) -> str:
    return f"m{mode}d{fault:02d}.mat"


def build_download_url(base_url: str, mode: int, fault: int) -> str:
    clean_base = base_url.rstrip("/") + "/"
    file_name = build_file_name(mode, fault)
    return f"{clean_base}M{mode}/{file_name}"


def iter_targets(modes: Iterable[int], faults: Iterable[int]) -> list[tuple[int, int]]:
    return [(mode, fault) for mode in modes for fault in faults]


def download_tep_file(
    session: requests.Session,
    base_url: str,
    save_folder: Path,
    mode: int,
    fault: int,
    timeout: int,
    retries: int,
    force: bool,
) -> tuple[str, str]:
    file_name = build_file_name(mode, fault)
    download_url = build_download_url(base_url, mode, fault)
    save_path = save_folder / file_name
    temp_path = save_path.with_suffix(save_path.suffix + ".part")
    target_label = format_target(mode, fault)

    if save_path.exists() and not force:
        return "skipped", f"[SKIP] {target_label} -> {file_name} already exists"

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(download_url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
            os.replace(temp_path, save_path)
            size_mb = save_path.stat().st_size / (1024 * 1024)
            return "success", f"[OK] {target_label} -> {file_name} ({size_mb:.2f} MB)"
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            if attempt < retries:
                print(f"[RETRY] {target_label} attempt {attempt}/{retries} failed, retrying...")

    return "failed", f"[FAIL] {target_label}, error: {last_error}"


def main() -> int:
    args = parse_args()
    save_folder = Path(args.save_folder).resolve()
    save_folder.mkdir(parents=True, exist_ok=True)

    targets = iter_targets(args.modes, args.faults)
    total = len(targets)
    success = 0
    skipped = 0
    failed = 0

    print("[START] Downloading selected TEP files...")
    print(f"BASE_URL: {args.base_url}")
    print(f"SAVE_FOLDER: {save_folder}")
    print(f"MODES: {args.modes} | FILE_FAULTS_DXX: {args.faults}")
    print(f"TOTAL_FILES: {total}")
    print("-" * 60)

    with requests.Session() as session:
        session.headers.update({"User-Agent": "CoReM-AD TEP downloader"})
        for mode, fault in targets:
            status, message = download_tep_file(
                session=session,
                base_url=args.base_url,
                save_folder=save_folder,
                mode=mode,
                fault=fault,
                timeout=args.timeout,
                retries=args.retries,
                force=args.force,
            )
            print(message)
            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

    print("-" * 60)
    print(f"[DONE] success={success} | skipped={skipped} | failed={failed}")
    print(f"DATA_DIR: {save_folder}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
