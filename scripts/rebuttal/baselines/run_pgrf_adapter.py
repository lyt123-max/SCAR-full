from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from common import (
    REPO_ROOT,
    load_common_data,
    measure_inference,
    save_standard_outputs,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCAR-side official PGRF-Net adapter.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--epochs-stage1", type=int, default=50)
    parser.add_argument("--epochs-stage2", type=int, default=20)
    parser.add_argument("--patience-stage1", type=int, default=10)
    parser.add_argument("--patience-stage2", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Formal PGRF-Net runs require seed 42.")
    official = REPO_ROOT / "third_party" / "baselines" / "PGRF-Net"
    sys.path.insert(0, str(official))
    import torch
    from inference import infer_scores
    from model import PGRFNet
    from training import train_model_stage1, train_model_stage2
    from utils import create_windows

    set_seed(args.seed)
    train, test, labels = load_common_data(args.data_dir)
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train).astype(np.float32)
    test = scaler.transform(test).astype(np.float32)
    train_labels = np.zeros(len(train), dtype=np.int8)
    x_train, y_train, l_train = create_windows(
        train, train_labels, args.window_size
    )
    if len(x_train) == 0:
        raise ValueError("Training sequence is shorter than PGRF-Net window size.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = PGRFNet(
        num_vars=x_train.shape[2],
        seq_len=args.window_size,
        num_protos=10,
        num_context_protos=10,
        num_spike_protos=10,
    ).to(device)
    params = {
        "epochs_stage1": args.epochs_stage1,
        "lr": 1e-4,
        "batch_size": 128,
        "patience_stage1": args.patience_stage1,
        "epochs_stage2": args.epochs_stage2,
        "lr_stage2": 1e-4,
        "patience_stage2": args.patience_stage2,
        "focal_gamma": 2.0,
        "focal_alpha": 0.5,
        "anomaly_weight": 10.0,
        "mask_reg_weight": 0.01,
        "mask_diff_weight": 0.001,
        "acyclic_penalty_weight": 1e-4,
        "lambda1": 1e-3,
        "sparsity_lambda": 1e-3,
        "context_loss_weight_stage1": 0.01,
        "spike_loss_weight_stage1": 0.01,
        "pseudo_normal_percent": 0.25,
        "gate_normal_suppress_weight": 0.1,
        "gate_entropy_weight": 0.001,
        "context_loss_weight_stage2": 0.2,
        "spike_loss_weight_stage2": 0.2,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(args.output_dir)
    try:
        train_model_stage1(model, x_train, y_train, l_train, **params)
        train_model_stage2(model, x_train, y_train, l_train, **params)
    finally:
        os.chdir(previous)
    torch.save(model.state_dict(), args.output_dir / "model.pt")

    def infer() -> np.ndarray:
        payload = infer_scores(model, test, args.window_size)
        predictive = payload["predictive_scores"]
        explanation = np.stack(
            [
                payload["structural_scores"],
                payload["contextual_scores"],
                payload["spike_scores"],
            ],
            axis=-1,
        )
        weights = explanation / (explanation.sum(axis=-1, keepdims=True) + 1e-8)
        return 0.9 * predictive + 0.1 * np.sum(weights * explanation, axis=-1)

    scores, timing = measure_inference(infer)
    save_standard_outputs(
        output_dir=args.output_dir,
        method="PGRF-Net",
        dataset=args.dataset,
        seed=args.seed,
        scores=scores,
        labels=labels,
        timing=timing,
        implementation={
            "upstream": "https://github.com/jahoonjeong/PGRF-Net.git",
            "commit": "5dc6f7522d20043eb31f6b2b13091c80ad394dcb",
            "adapter": "official model/training/inference with project-side paths",
        },
    )


if __name__ == "__main__":
    main()
