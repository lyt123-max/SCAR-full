from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad import CoReMADConfig, CoReMADTrainer
from coremad.data import build_loader
from scripts.rebuttal.muqn.proxy_metrics import (
    extract_state_proxy_features,
    normalize_proxy_features,
)


STRATEGIES = {
    "full": (True, True),
    "no_state": (False, True),
    "context_only": (False, True),
    "no_context": (True, False),
    "state_only": (True, False),
    "global": (False, False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export retrieval provenance for Reviewer muQn experiments."
    )
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=sorted(STRATEGIES),
        default=list(STRATEGIES),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--aggregate_top_k", type=int, default=20)
    return parser.parse_args()


def _load_context(experiment_dir: Path, device: str | None) -> dict:
    config = CoReMADConfig.load(experiment_dir / "config.json")
    config.artifact_root = str(experiment_dir.parent)
    config.experiment_name = experiment_dir.name
    if device is not None:
        config.device = device
    trainer = CoReMADTrainer(config)
    model, normalizer, _ = trainer.load_stage_a_model()
    memory = trainer.load_memory_bank()
    raw_bundle = trainer._load_raw_bundle()
    bundle = trainer.transform_bundle_with_normalizer(raw_bundle, normalizer)
    return {
        "config": config,
        "trainer": trainer,
        "model": model,
        "memory": memory,
        "bundle": bundle,
    }


def _window_batch(data: np.ndarray, starts: np.ndarray, seq_len: int) -> np.ndarray:
    valid = (starts >= 0) & (starts + seq_len <= len(data))
    if not np.all(valid):
        invalid = starts[~valid][:5].tolist()
        raise ValueError(f"Window starts fall outside source data: {invalid}")
    return np.stack([data[int(start) : int(start) + seq_len] for start in starts], axis=0)


def _aggregate_neighbors(
    window_ids_by_scale: list[torch.Tensor],
    raw_starts_by_scale: list[torch.Tensor],
    distances_by_scale: list[torch.Tensor],
    valid_by_scale: list[torch.Tensor],
    row: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best: dict[int, tuple[float, int]] = {}
    for ids, raw_starts, distances, valid in zip(
        window_ids_by_scale,
        raw_starts_by_scale,
        distances_by_scale,
        valid_by_scale,
    ):
        ids_np = ids[row].detach().cpu().numpy().reshape(-1)
        starts_np = raw_starts[row].detach().cpu().numpy().reshape(-1)
        distances_np = distances[row].detach().cpu().numpy().reshape(-1)
        valid_np = valid[row].detach().cpu().numpy().reshape(-1)
        for window_id, raw_start, distance, is_valid in zip(
            ids_np, starts_np, distances_np, valid_np
        ):
            if not is_valid or int(window_id) < 0 or not np.isfinite(distance):
                continue
            current = best.get(int(window_id))
            if current is None or float(distance) < current[0]:
                best[int(window_id)] = (float(distance), int(raw_start))
    ordered = sorted(best.items(), key=lambda item: item[1][0])[:top_k]
    window_ids = np.full(top_k, -1, dtype=np.int64)
    raw_starts = np.full(top_k, -1, dtype=np.int64)
    distances = np.full(top_k, np.inf, dtype=np.float32)
    for index, (window_id, (distance, raw_start)) in enumerate(ordered):
        window_ids[index] = window_id
        raw_starts[index] = raw_start
        distances[index] = distance
    return window_ids, raw_starts, distances


def _export_strategy(
    strategy: str,
    ctx: dict,
    loader,
    output_path: Path,
    aggregate_top_k: int,
) -> int:
    config = ctx["config"]
    model = ctx["model"]
    memory = ctx["memory"]
    bundle = ctx["bundle"]
    use_two_level, use_context = STRATEGIES[strategy]
    memory.config.use_two_level_retrieval = use_two_level
    memory.config.use_context_key_retrieval = use_context

    state_starts = memory.state_window_starts.detach().cpu().numpy().astype(np.int64)
    state_windows = _window_batch(bundle.train_full, state_starts, config.seq_len)
    state_proxy = extract_state_proxy_features(state_windows)
    normalized_state_proxy, proxy_stats = normalize_proxy_features(state_proxy, state_proxy)

    records: dict[str, list[np.ndarray | int]] = {
        "query_start": [],
        "query_label": [],
        "query_proxy": [],
        "coarse_window_ids": [],
        "coarse_window_starts": [],
        "coarse_distances": [],
        "neighbor_window_ids": [],
        "neighbor_window_starts": [],
        "neighbor_raw_starts": [],
        "neighbor_context_distances": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(ctx["trainer"].device, non_blocking=True)
            encoded = model.encode(x)
            details = memory.query_preencoded(
                encoded["state_vec"],
                encoded["z"],
                encoded["c"],
                return_details=True,
            )
            starts = batch["start"].detach().cpu().numpy().astype(np.int64)
            query_proxy = extract_state_proxy_features(x.detach().cpu().numpy())
            query_proxy, _ = normalize_proxy_features(state_proxy, query_proxy)
            labels = batch.get("labels")
            if labels is None:
                query_labels = np.full(len(starts), -1, dtype=np.int8)
            else:
                query_labels = (
                    labels.detach().cpu().numpy().reshape(len(starts), -1).max(axis=1) > 0
                ).astype(np.int8)

            coarse_ids = np.asarray(details["coarse_windows"], dtype=np.int64)
            coarse_starts = details["coarse_window_starts"].detach().cpu().numpy().astype(np.int64)
            coarse_distances = details["coarse_distances"].detach().cpu().numpy()
            for row, start in enumerate(starts.tolist()):
                neighbor_ids, neighbor_raw_starts, neighbor_distances = _aggregate_neighbors(
                    details["neighbor_window_ids"],
                    details["neighbor_raw_starts"],
                    details["neighbor_context_distances"],
                    details["neighbor_valid_masks"],
                    row,
                    aggregate_top_k,
                )
                neighbor_starts = np.full(aggregate_top_k, -1, dtype=np.int64)
                valid_ids = neighbor_ids >= 0
                neighbor_starts[valid_ids] = state_starts[neighbor_ids[valid_ids]]
                records["query_start"].append(int(start))
                records["query_label"].append(int(query_labels[row]))
                records["query_proxy"].append(query_proxy[row].astype(np.float32))
                records["coarse_window_ids"].append(coarse_ids[row])
                records["coarse_window_starts"].append(coarse_starts[row])
                records["coarse_distances"].append(coarse_distances[row])
                records["neighbor_window_ids"].append(neighbor_ids)
                records["neighbor_window_starts"].append(neighbor_starts)
                records["neighbor_raw_starts"].append(neighbor_raw_starts)
                records["neighbor_context_distances"].append(neighbor_distances)

    payload = {key: np.asarray(values) for key, values in records.items()}
    payload.update(
        {
            "schema_version": np.asarray(1, dtype=np.int32),
            "strategy": np.asarray(strategy),
            "dataset": np.asarray(config.dataset),
            "seq_len": np.asarray(config.seq_len, dtype=np.int32),
            "proxy_center": proxy_stats["center"].astype(np.float32),
            "proxy_scale": proxy_stats["scale"].astype(np.float32),
            "state_window_starts": state_starts,
            "state_proxy": normalized_state_proxy.astype(np.float32),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return len(records["query_start"])


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    ctx = _load_context(experiment_dir, args.device)
    config = ctx["config"]
    stride = config.test_stride if args.stride is None else int(args.stride)
    loader = build_loader(
        data=ctx["bundle"].test,
        labels=ctx["bundle"].test_labels,
        seq_len=config.seq_len,
        stride=stride,
        batch_size=config.test_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        max_windows=args.max_windows,
        drop_last=False,
        segment_ranges=ctx["bundle"].test_segment_ranges,
    )
    output_dir = args.output_dir or (experiment_dir / "rebuttal_muqn" / "retrieval")
    manifest = {
        "experiment_dir": str(experiment_dir),
        "dataset": config.dataset,
        "stride": stride,
        "max_windows": int(args.max_windows),
        "aggregate_top_k": int(args.aggregate_top_k),
        "strategies": {},
    }
    for strategy in args.strategies:
        path = output_dir / f"retrieval_{strategy}.npz"
        count = _export_strategy(strategy, ctx, loader, path, args.aggregate_top_k)
        manifest["strategies"][strategy] = {"path": str(path), "num_windows": count}
        print(f"[muQn] {strategy}: exported {count} windows to {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
