from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    REPO_ROOT,
    load_common_data,
    measure_inference,
    overlap_average,
    save_standard_outputs,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCAR-side official CATCH adapter.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _point_scores(detector, test_frame: pd.DataFrame) -> np.ndarray:
    import torch
    import torch.nn as nn

    from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion
    from ts_benchmark.baselines.utils import anomaly_detection_data_provider

    transformed = pd.DataFrame(
        detector.scaler.transform(test_frame.values),
        columns=test_frame.columns,
        index=test_frame.index,
    )
    detector.model.load_state_dict(detector.early_stopping.check_point)
    loader = anomaly_detection_data_provider(
        transformed,
        batch_size=detector.config.batch_size,
        win_size=detector.config.seq_len,
        step=1,
        mode="thre",
    )
    temporal = nn.MSELoss(reduction="none")
    frequency = frequency_criterion(detector.config)
    rows = []
    detector.model.eval()
    with torch.inference_mode():
        for batch, _ in loader:
            batch = batch.float().to(detector.device)
            output, _, _ = detector.model(batch)
            score = torch.mean(temporal(batch, output), dim=-1)
            score += detector.config.score_lambda * torch.mean(
                frequency(batch, output), dim=-1
            )
            rows.append(score.detach().cpu().numpy())
    return overlap_average(np.concatenate(rows, axis=0), len(test_frame))


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Formal CATCH runs require seed 42.")
    official = REPO_ROOT / "third_party" / "baselines" / "CATCH"
    sys.path.insert(0, str(official))
    from ts_benchmark.baselines.catch.CATCH import CATCH

    train, test, labels = load_common_data(args.data_dir)
    set_seed(args.seed)
    train_frame = pd.DataFrame(train)
    test_frame = pd.DataFrame(test)
    detector = CATCH(seq_len=128, patch_size=16, inference_patch_size=32)
    detector.detect_fit(train_frame, test_frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(detector.model.state_dict(), args.output_dir / "model.pt")
    scores, timing = measure_inference(lambda: _point_scores(detector, test_frame))
    save_standard_outputs(
        output_dir=args.output_dir,
        method="CATCH",
        dataset=args.dataset,
        seed=args.seed,
        scores=scores,
        labels=labels,
        timing=timing,
        implementation={
            "upstream": "https://github.com/decisionintelligence/CATCH.git",
            "commit": "3647c69be5eb56649b072596cf89098e689e20c3",
            "adapter": "official CATCH class with overlap-averaged point export",
        },
    )


if __name__ == "__main__":
    main()
