from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.budget import select_budget_candidate
from scripts.experiments.match_budget import _bank_bytes, _parameter_count, _time_top_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize the five-dataset lightweight E31 budget match."
    )
    parser.add_argument("--target-experiment", type=Path, required=True)
    parser.add_argument("--base-experiment", type=Path, required=True)
    parser.add_argument(
        "--q-experiment",
        action="append",
        default=[],
        metavar="RATIO=PATH",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.15)
    return parser.parse_args()


def parse_q_experiments(
    base_experiment: Path,
    values: list[str],
) -> list[tuple[float, Path]]:
    rows = [(1.0, base_experiment.resolve())]
    for value in values:
        ratio_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Invalid --q-experiment value: {value!r}.")
        ratio = float(ratio_text)
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError("Lightweight q ratios must be in the open interval (0, 1).")
        rows.append((ratio, Path(path_text).resolve()))
    if len({ratio for ratio, _ in rows}) != len(rows):
        raise ValueError("Duplicate q ratios are not allowed.")
    return rows


def main() -> None:
    args = parse_args()
    target = args.target_experiment.resolve()
    base = args.base_experiment.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    target_parameters = _parameter_count(target)
    base_parameters = _parameter_count(base)
    parameter_error = abs(base_parameters - target_parameters) / target_parameters
    if parameter_error > float(args.tolerance):
        raise ValueError(
            "The lightweight E31 L512 base is outside the registered parameter "
            f"tolerance: {parameter_error:.6f} > {float(args.tolerance):.6f}."
        )
    q_rows = [
        {
            "keep_ratio": ratio,
            "bank_bytes": _bank_bytes(experiment),
            "experiment": str(experiment),
        }
        for ratio, experiment in parse_q_experiments(base, args.q_experiment)
    ]
    target_bank = _bank_bytes(target)
    selected_q = min(
        q_rows,
        key=lambda row: (
            abs(float(row["bank_bytes"]) - target_bank) / target_bank,
            float(row["bank_bytes"]),
            float(row["keep_ratio"]),
        ),
    )
    selected_experiment = Path(str(selected_q["experiment"]))

    target_timing_path = (
        target.parent / f"scar_efficiency_{target.name.removeprefix('scar_main_').removesuffix('_seed42')}_seed42"
        / "timing.json"
    )
    if not target_timing_path.is_file():
        raise FileNotFoundError(
            "Lightweight E31 requires the registered SCAR timing artifact: "
            f"{target_timing_path}"
        )
    target_timing = json.loads(target_timing_path.read_text(encoding="utf-8"))
    target_latency_ms = float(target_timing["mean_seconds"]) * 1000.0
    candidates = []
    for top_k in args.top_k:
        latency_ms = _time_top_k(selected_experiment, int(top_k))
        candidates.append(
            {
                "name": f"lite_q{selected_q['keep_ratio']}_k{top_k}",
                "d_z": int(
                    json.loads((base / "config.json").read_text(encoding="utf-8"))[
                        "d_z"
                    ]
                ),
                "keep_ratio": float(selected_q["keep_ratio"]),
                "top_K": int(top_k),
                "parameters": int(base_parameters),
                "bank_bytes": int(selected_q["bank_bytes"]),
                "latency_ms": float(latency_ms),
                "config_size": int(top_k),
            }
        )
    result = select_budget_candidate(
        {
            "parameters": target_parameters,
            "bank_bytes": target_bank,
            "latency_ms": target_latency_ms,
        },
        candidates,
        tolerance=float(args.tolerance),
    )
    result["lightweight_protocol"] = {
        "base_experiment": str(base),
        "parameter_relative_error": parameter_error,
        "q_candidates": q_rows,
        "selected_q": selected_q,
        "top_k_candidates": [int(value) for value in args.top_k],
        "warmup_runs": 1,
        "timed_runs": 3,
        "selection_uses_test_performance": False,
    }
    (output / "budget_match.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output / "budget_match.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    (output / "parameter_selection.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_parameters": target_parameters,
                "base_parameters": base_parameters,
                "relative_error": parameter_error,
                "base_experiment": str(base),
                "selection_rule": "reuse E29 L512 when within registered tolerance",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "rebuttal"
                / "muqn"
                / "score_retrieval_strategies.py"
            ),
            "--experiment-dir",
            str(selected_experiment),
            "--strategy",
            "global",
            "--top-k",
            str(result["candidate"]["top_K"]),
            "--output-dir",
            str(output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
