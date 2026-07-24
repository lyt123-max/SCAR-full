from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rebuttal.muqn.proxy_metrics import (
    block_bootstrap_mean,
    jaccard_index,
    lagged_jaccard,
    paired_proxy_distances,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute condition compatibility and retrieval stability from muQn logs."
    )
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument(
        "--log-files",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit strategy NPZ files; useful when each strategy has its own run directory.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 4, 16])
    return parser.parse_args()


def _paired_proxy_from_ids(payload, id_key: str) -> dict[str, np.ndarray]:
    query = np.asarray(payload["query_proxy"], dtype=np.float64)
    ids = np.asarray(payload[id_key], dtype=np.int64)
    state_proxy = np.asarray(payload["state_proxy"], dtype=np.float64)
    query_rows = []
    neighbor_rows = []
    for row in range(ids.shape[0]):
        for neighbor_id in ids[row]:
            if int(neighbor_id) < 0:
                continue
            query_rows.append(query[row])
            neighbor_rows.append(state_proxy[int(neighbor_id)])
    if not query_rows:
        return {
            "proxy_l1": np.empty(0, dtype=np.float64),
            "proxy_correlation_distance": np.empty(0, dtype=np.float64),
        }
    return paired_proxy_distances(np.asarray(query_rows), np.asarray(neighbor_rows))


def _summarize(
    values: np.ndarray,
    block_size: int,
    n_bootstrap: int,
    seed: int,
) -> dict:
    return block_bootstrap_mean(
        values,
        block_size=block_size,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    output_dir = args.output_dir or log_dir
    paths = (
        [path.resolve() for path in args.log_files]
        if args.log_files
        else sorted(log_dir.glob("retrieval_*.npz"))
    )
    if not paths:
        raise FileNotFoundError(f"No retrieval_*.npz files found in {log_dir}.")

    payloads = {}
    for path in paths:
        payload = np.load(path, allow_pickle=False)
        strategy = str(np.asarray(payload["strategy"]).item())
        payloads[strategy] = payload

    summaries: dict[str, dict] = {}
    rows: list[dict[str, object]] = []
    try:
        for strategy, payload in payloads.items():
            strategy_summary: dict[str, dict] = {}
            for level, id_key in (
                ("coarse", "coarse_window_ids"),
                ("fine", "neighbor_window_ids"),
            ):
                metrics = _paired_proxy_from_ids(payload, id_key)
                for metric_name, values in metrics.items():
                    key = f"{level}_{metric_name}"
                    result = _summarize(
                        values,
                        args.block_size,
                        args.n_bootstrap,
                        args.seed,
                    )
                    strategy_summary[key] = result
                    rows.append({"strategy": strategy, "metric": key, **result})

            temporal: dict[str, dict] = {}
            neighbor_sets = np.asarray(payload["neighbor_window_ids"], dtype=np.int64)
            for lag in args.lags:
                if lag >= len(neighbor_sets):
                    continue
                values = lagged_jaccard(neighbor_sets, lag)
                result = _summarize(
                    values,
                    args.block_size,
                    args.n_bootstrap,
                    args.seed + int(lag),
                )
                temporal[str(lag)] = result
                rows.append(
                    {
                        "strategy": strategy,
                        "metric": f"temporal_jaccard_lag_{lag}",
                        **result,
                    }
                )
            strategy_summary["temporal_jaccard"] = temporal
            summaries[strategy] = strategy_summary

        if "full" in payloads:
            full_starts = np.asarray(payloads["full"]["query_start"])
            full_sets = np.asarray(payloads["full"]["neighbor_window_ids"])
            for strategy, payload in payloads.items():
                if strategy == "full":
                    continue
                starts = np.asarray(payload["query_start"])
                if not np.array_equal(starts, full_starts):
                    raise ValueError(
                        f"Strategy {strategy} does not align with full query starts."
                    )
                other_sets = np.asarray(payload["neighbor_window_ids"])
                values = np.asarray(
                    [
                        jaccard_index(full_sets[row], other_sets[row])
                        for row in range(len(full_sets))
                    ],
                    dtype=np.float64,
                )
                result = _summarize(
                    values,
                    args.block_size,
                    args.n_bootstrap,
                    args.seed,
                )
                summaries[strategy]["jaccard_vs_full"] = result
                rows.append(
                    {
                        "strategy": strategy,
                        "metric": "jaccard_vs_full",
                        **result,
                    }
                )
    finally:
        for payload in payloads.values():
            payload.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_evidence.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "block_size": args.block_size,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "summaries": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    csv_path = output_dir / "retrieval_evidence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "metric",
                "mean",
                "ci_low",
                "ci_high",
                "n",
                "block_size",
                "n_bootstrap",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[muQn] wrote {json_path}")
    print(f"[muQn] wrote {csv_path}")


if __name__ == "__main__":
    main()
