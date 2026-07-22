from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coremad.data import build_loader
from tep_common import (
    aggregate_neighbor_lists,
    aggregate_point_scores,
    get_loader_starts,
    load_experiment_context,
    metadata_for_starts,
    np_to_python,
    parse_tep_name,
)


WINDOW_SCORE_KEYS = [
    "memory_distance",
    "soft_support_score",
    "state_novelty",
    "completion_scale8",
    "completion_scale32",
    "raw_max",
    "zscore_mean",
    "cdf_max",
    "cdf_mean",
    "cdf_mean_soft_support",
    "cdf_softmax",
    "final",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export offline TEP mechanism logs.")
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--window_aggregation",
        type=str,
        default="p95",
        choices=["p95", "top5_mean", "max", "mean"],
    )
    parser.add_argument("--top_k", type=int, default=0, help="Defaults to config.top_K.")
    return parser.parse_args()


def _build_sequence_records(sequence_scores_npz: Path) -> list[dict[str, object]]:
    payload = np.load(sequence_scores_npz, allow_pickle=True)
    score_keys = [key for key in payload.files if key not in {"labels", "names", "covered", "coverage_ratio"}]
    names = payload["names"].tolist()
    labels = np.asarray(payload["labels"], dtype=np.int32)
    covered = np.asarray(payload["covered"], dtype=bool)
    coverage_ratio = np.asarray(payload["coverage_ratio"], dtype=np.float64)
    records: list[dict[str, object]] = []
    for idx, name in enumerate(names):
        meta = parse_tep_name(str(name))
        record = {
            "sequence_name": str(name),
            "label": int(labels[idx]),
            "covered": bool(covered[idx]) if idx < len(covered) else False,
            "coverage_ratio": float(coverage_ratio[idx]) if idx < len(coverage_ratio) else float("nan"),
            "mode_id": int(meta["mode_id"]),
            "fault_id": int(meta["fault_id"]),
            "is_normal": int(meta["is_normal"]),
            "file_id": str(meta["file_id"]),
        }
        for score_key in score_keys:
            record[score_key] = float(np.asarray(payload[score_key], dtype=np.float64)[idx])
        if "knn_distance" in record and "memory_distance" not in record:
            record["memory_distance"] = float(record["knn_distance"])
        records.append(record)
    return records


def _save_window_records(output_dir: Path, prefix: str, records: list[dict[str, object]]) -> None:
    if not records:
        raise RuntimeError(f"No records collected for prefix={prefix}.")

    np_payload: dict[str, np.ndarray] = {
        "window_index": np.asarray([record["window_index"] for record in records], dtype=np.int64),
        "start": np.asarray([record["start"] for record in records], dtype=np.int64),
        "mode_id": np.asarray([record["mode_id"] for record in records], dtype=np.int32),
        "file_id": np.asarray([record["file_id"] for record in records], dtype=object),
        "sequence_name": np.asarray([record["sequence_name"] for record in records], dtype=object),
        "is_normal": np.asarray([record["is_normal"] for record in records], dtype=np.int32),
        "fault_id": np.asarray([record["fault_id"] for record in records], dtype=np.int32),
        "state_vec": np.stack([np.asarray(record["state_vec"], dtype=np.float32) for record in records], axis=0),
        "topk_neighbor_window_ids": np.stack(
            [np.asarray(record["topk_neighbor_window_ids"], dtype=np.int64) for record in records],
            axis=0,
        ),
        "topk_neighbor_mode_ids": np.stack(
            [np.asarray(record["topk_neighbor_mode_ids"], dtype=np.int32) for record in records],
            axis=0,
        ),
        "topk_neighbor_file_ids": np.stack(
            [np.asarray(record["topk_neighbor_file_ids"], dtype=object) for record in records],
            axis=0,
        ),
        "topk_neighbor_distances": np.stack(
            [np.asarray(record["topk_neighbor_distances"], dtype=np.float64) for record in records],
            axis=0,
        ),
    }
    for score_key in WINDOW_SCORE_KEYS:
        np_payload[score_key] = np.asarray([record[score_key] for record in records], dtype=np.float64)

    np.savez(output_dir / f"{prefix}.npz", **np_payload)

    csv_fields = [
        "window_index",
        "start",
        "mode_id",
        "file_id",
        "sequence_name",
        "is_normal",
        "fault_id",
        *WINDOW_SCORE_KEYS,
        "state_vec",
        "topk_neighbor_window_ids",
        "topk_neighbor_mode_ids",
        "topk_neighbor_file_ids",
        "topk_neighbor_distances",
    ]
    csv_path = output_dir / f"{prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {}
            for key in csv_fields:
                value = record[key]
                row[key] = json.dumps(np_to_python(value), ensure_ascii=False) if isinstance(value, np.ndarray) else value
            writer.writerow(row)
    print(f"[TEP Export] wrote {prefix} npz: {output_dir / f'{prefix}.npz'}")
    print(f"[TEP Export] wrote {prefix} csv: {csv_path}")


def _collect_window_records(
    trainer,
    model,
    memory,
    cdf_fusion,
    zscore_fusion,
    loader,
    metadata_for_start: Callable[[int], dict[str, object]],
    train_meta: dict[str, np.ndarray],
    window_aggregation: str,
    top_k: int,
    record_offset: int = 0,
) -> list[dict[str, object]]:
    model.eval()
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(trainer.device)
            starts = batch["start"].detach().cpu().numpy().astype(np.int64)

            if trainer._completion_score_names():
                encoded, comp_scores = model.deterministic_completion_scores(x)
            else:
                encoded = model.encode(x)
                comp_scores = []
            memory_out = memory.query_preencoded(
                encoded["state_vec"],
                encoded["z"],
                encoded["c"],
                return_details=True,
            )

            point_diags: dict[str, np.ndarray] = {
                "knn_distance": trainer._fuse_patch_scores(memory_out["mem_scores"]).detach().cpu().numpy(),
                "state_novelty": memory_out["state_novelty"].to(x.device).unsqueeze(-1).expand(-1, trainer.config.seq_len).detach().cpu().numpy(),
            }
            if trainer.config.use_prototype_support:
                point_diags["soft_support_score"] = (
                    trainer._fuse_patch_scores(memory_out["soft_mem_scores"]).detach().cpu().numpy()
                )
            for score_name, patch_size, comp_score in zip(
                trainer._completion_score_names(),
                trainer.config.patch_sizes,
                comp_scores,
            ):
                point_diags[score_name] = trainer._expand_patch_score(comp_score, patch_size).detach().cpu().numpy()

            _, _, metric_sources, _ = trainer._build_metric_sources_from_point_diags(
                point_diags,
                cdf_fusion,
                zscore_fusion,
            )

            for row_idx, start in enumerate(starts.tolist()):
                meta = metadata_for_start(int(start))
                neighbor_ids, neighbor_dists = aggregate_neighbor_lists(
                    neighbor_ids=[tensor[row_idx].detach().cpu().numpy() for tensor in memory_out["neighbor_window_ids"]],
                    neighbor_dists=[tensor[row_idx].detach().cpu().numpy() for tensor in memory_out["neighbor_context_distances"]],
                    neighbor_valid=[tensor[row_idx].detach().cpu().numpy() for tensor in memory_out["neighbor_valid_masks"]],
                    top_k=top_k,
                )
                neighbor_mode_ids = np.full(top_k, -1, dtype=np.int32)
                neighbor_file_ids = np.empty(top_k, dtype=object)
                neighbor_file_ids[:] = ""
                for neigh_idx, window_id in enumerate(neighbor_ids.tolist()):
                    if window_id < 0:
                        continue
                    neighbor_mode_ids[neigh_idx] = int(train_meta["mode_id"][window_id])
                    neighbor_file_ids[neigh_idx] = str(train_meta["file_id"][window_id])

                record = {
                    "window_index": int(record_offset + len(records)),
                    "start": int(start),
                    "mode_id": int(meta["mode_id"]),
                    "file_id": str(meta["file_id"]),
                    "sequence_name": str(meta["sequence_name"]),
                    "is_normal": int(meta["is_normal"]),
                    "fault_id": int(meta["fault_id"]),
                    "state_vec": encoded["state_vec"][row_idx].detach().cpu().numpy().astype(np.float32),
                    "memory_distance": aggregate_point_scores(point_diags["knn_distance"][row_idx], window_aggregation),
                    "soft_support_score": aggregate_point_scores(
                        point_diags["soft_support_score"][row_idx],
                        window_aggregation,
                    )
                    if "soft_support_score" in point_diags
                    else float("nan"),
                    "state_novelty": aggregate_point_scores(point_diags["state_novelty"][row_idx], window_aggregation),
                    "completion_scale8": float("nan"),
                    "completion_scale32": float("nan"),
                    "raw_max": aggregate_point_scores(metric_sources["raw_max"][row_idx], window_aggregation),
                    "zscore_mean": aggregate_point_scores(metric_sources["zscore_mean"][row_idx], window_aggregation),
                    "cdf_max": aggregate_point_scores(metric_sources["cdf_max"][row_idx], window_aggregation),
                    "cdf_mean": aggregate_point_scores(metric_sources["cdf_mean"][row_idx], window_aggregation),
                    "cdf_mean_soft_support": aggregate_point_scores(
                        metric_sources["cdf_mean_soft_support"][row_idx],
                        window_aggregation,
                    ),
                    "cdf_softmax": aggregate_point_scores(metric_sources["cdf_softmax"][row_idx], window_aggregation),
                    "final": aggregate_point_scores(metric_sources["selected"][row_idx], window_aggregation),
                    "topk_neighbor_window_ids": neighbor_ids.astype(np.int64),
                    "topk_neighbor_mode_ids": neighbor_mode_ids.astype(np.int32),
                    "topk_neighbor_file_ids": neighbor_file_ids.astype(object),
                    "topk_neighbor_distances": neighbor_dists.astype(np.float64),
                }
                for score_name in trainer._completion_score_names():
                    record[score_name] = aggregate_point_scores(point_diags[score_name][row_idx], window_aggregation)
                records.append(record)
    return records


def main() -> None:
    args = parse_args()
    ctx = load_experiment_context(args.experiment_dir, device=args.device)
    config = ctx["config"]
    if str(config.dataset).upper() != "TEP":
        raise ValueError(f"TEP mechanism export only supports dataset=TEP, got {config.dataset!r}")

    output_dir = args.output_dir or (args.experiment_dir / "tep_mechanism")
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k = int(args.top_k) if int(args.top_k) > 0 else int(config.top_K)

    trainer = ctx["trainer"]
    model = ctx["model"]
    normalizer = ctx["normalizer"]
    memory = ctx["memory"]
    raw_bundle = ctx["raw_bundle"]
    cdf_fusion = ctx["cdf_fusion"]
    zscore_fusion = ctx["zscore_fusion"]
    model.eval()

    if raw_bundle.val is None or raw_bundle.val_segment_ranges is None or raw_bundle.val_segment_names is None:
        raise RuntimeError("TEP mechanism export requires validation normal sequences with segment metadata.")
    if raw_bundle.test_sequences is None or raw_bundle.test_sequence_names is None:
        raise RuntimeError("TEP mechanism export requires fault test sequences.")

    train_starts = get_loader_starts(
        data=raw_bundle.train,
        config=config,
        stride=config.memory_build_stride,
        max_windows=config.max_train_windows,
        segment_ranges=raw_bundle.train_segment_ranges,
    )
    if len(train_starts) != int(memory.state_bank.size(0)):
        raise RuntimeError(
            f"State bank/window count mismatch: starts={len(train_starts)} state_bank={int(memory.state_bank.size(0))}"
        )
    train_meta = metadata_for_starts(
        starts=train_starts,
        ranges=raw_bundle.train_segment_ranges,
        names=raw_bundle.train_segment_names or [],
    )
    prototype_labels = memory.prototype_labels.cpu().numpy() if int(memory.prototype_labels.numel()) == len(train_starts) else np.full(
        len(train_starts),
        -1,
        dtype=np.int32,
    )
    np.savez(
        output_dir / "train_state_meta.npz",
        window_id=np.arange(len(train_starts), dtype=np.int64),
        start=train_starts.astype(np.int64),
        mode_id=train_meta["mode_id"].astype(np.int32),
        file_id=train_meta["file_id"].astype(object),
        prototype_label=np.asarray(prototype_labels, dtype=np.int32),
    )
    print(f"[TEP Export] wrote train state meta: {output_dir / 'train_state_meta.npz'}")

    audit_norm = normalizer.transform(raw_bundle.val)
    audit_loader = build_loader(
        data=audit_norm,
        labels=None,
        seq_len=config.seq_len,
        stride=config.test_stride,
        batch_size=config.test_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        max_windows=0,
        drop_last=False,
        segment_ranges=raw_bundle.val_segment_ranges,
    )
    audit_starts = np.asarray(audit_loader.dataset.start_indices, dtype=np.int64).reshape(-1)
    audit_meta = metadata_for_starts(
        starts=audit_starts,
        ranges=raw_bundle.val_segment_ranges,
        names=raw_bundle.val_segment_names,
    )
    audit_meta_lookup = {int(start): idx for idx, start in enumerate(audit_starts.tolist())}
    audit_records = _collect_window_records(
        trainer=trainer,
        model=model,
        memory=memory,
        cdf_fusion=cdf_fusion,
        zscore_fusion=zscore_fusion,
        loader=audit_loader,
        metadata_for_start=lambda start: {
            "mode_id": int(audit_meta["mode_id"][audit_meta_lookup[start]]),
            "file_id": str(audit_meta["file_id"][audit_meta_lookup[start]]),
            "sequence_name": str(audit_meta["name"][audit_meta_lookup[start]]),
            "is_normal": int(audit_meta["is_normal"][audit_meta_lookup[start]]),
            "fault_id": int(audit_meta["fault_id"][audit_meta_lookup[start]]),
        },
        train_meta=train_meta,
        window_aggregation=args.window_aggregation,
        top_k=top_k,
        record_offset=0,
    )
    _save_window_records(output_dir, "audit_normal_window_logs", audit_records)

    fault_records: list[dict[str, object]] = []
    for sequence_name, sequence_data in zip(raw_bundle.test_sequence_names, raw_bundle.test_sequences):
        seq_meta = parse_tep_name(str(sequence_name))
        seq_norm = normalizer.transform(np.asarray(sequence_data, dtype=np.float32))
        seq_loader = build_loader(
            data=seq_norm,
            labels=None,
            seq_len=config.seq_len,
            stride=config.test_stride,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            max_windows=0,
            drop_last=False,
            segment_ranges=None,
        )
        seq_records = _collect_window_records(
            trainer=trainer,
            model=model,
            memory=memory,
            cdf_fusion=cdf_fusion,
            zscore_fusion=zscore_fusion,
            loader=seq_loader,
            metadata_for_start=lambda start, meta=seq_meta: {
                "mode_id": int(meta["mode_id"]),
                "file_id": str(meta["file_id"]),
                "sequence_name": str(meta["name"]),
                "is_normal": int(meta["is_normal"]),
                "fault_id": int(meta["fault_id"]),
            },
            train_meta=train_meta,
            window_aggregation=args.window_aggregation,
            top_k=top_k,
            record_offset=len(fault_records),
        )
        fault_records.extend(seq_records)
    _save_window_records(output_dir, "fault_window_logs", fault_records)

    fault_sequence_records = _build_sequence_records(args.experiment_dir / "test_sequence_scores.npz")
    (output_dir / "fault_sequence_scores_with_meta.json").write_text(
        json.dumps(fault_sequence_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "sequence_scores_with_meta.json").write_text(
        json.dumps(fault_sequence_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    meta = {
        "experiment_dir": str(args.experiment_dir),
        "output_dir": str(output_dir),
        "dataset": str(config.dataset),
        "window_aggregation": args.window_aggregation,
        "top_k": top_k,
        "n_train_state_windows": len(train_starts),
        "n_audit_normal_windows": len(audit_records),
        "n_fault_windows": len(fault_records),
        "n_fault_sequences": len(fault_sequence_records),
        "score_key": str(config.evaluation_score_key),
        "schema_version": 2,
    }
    (output_dir / "export_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[TEP Export] wrote fault sequence meta json: {output_dir / 'fault_sequence_scores_with_meta.json'}")


if __name__ == "__main__":
    main()
