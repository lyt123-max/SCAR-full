from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.preprocessing import StandardScaler

from common import (
    REPO_ROOT,
    load_common_data,
    measure_inference,
    overlap_average,
    save_standard_outputs,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCAR-side PUAD architecture adapter.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _windows(torch, values: np.ndarray, width: int, step: int):
    starts = range(0, len(values) - width + 1, step)
    rows = [values[start : start + width] for start in starts]
    if not rows:
        raise ValueError("Series is shorter than the PUAD window size.")
    return torch.from_numpy(np.stack(rows))


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Formal PUAD runs require seed 42.")
    official = REPO_ROOT / "third_party" / "baselines" / "PUAD"
    sys.path.insert(0, str(official))
    import torch
    import torch.nn.functional as functional
    from model10.AnomalyTransformer import AnomalyTransformer

    set_seed(args.seed)
    train, test, labels = load_common_data(args.data_dir)
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train).astype(np.float32)
    test = scaler.transform(test).astype(np.float32)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_args = SimpleNamespace(
        T=args.window_size,
        d_model=512,
        d_model_concept=256,
        d_vae=512,
        d_c=256,
        n_concepts=10,
        n_privately_concepts=2,
        softmax_t=1.0,
        lam=10.0,
    )
    model = AnomalyTransformer(
        args=model_args,
        win_size=args.window_size,
        enc_in=train.shape[1],
        c_out=train.shape[1],
        d_model=512,
        e_layers=3,
        device=device,
        model_state="train",
    ).to(device)
    train_windows = _windows(torch, train, args.window_size, args.window_size)
    loader = torch.utils.data.DataLoader(
        train_windows, batch_size=args.batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
    model.train()
    for epoch in range(args.epochs):
        warmup = min(epoch / 20.0, 1.0)
        for batch in loader:
            batch = batch.float().to(device)
            output, _, ot_loss, mse_loss, kl_c, kl_z, _ = model(batch)
            loss = functional.mse_loss(output, batch)
            loss += warmup * 0.01 * (kl_c + kl_z)
            loss += 0.01 * ot_loss.mean() + 0.0 * mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")

    def infer() -> np.ndarray:
        windows = _windows(torch, test, args.window_size, 1)
        batches = torch.utils.data.DataLoader(
            windows, batch_size=args.batch_size, shuffle=False
        )
        rows = []
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                batch = batch.float().to(device)
                output, *_ = model(batch)
                rows.append(
                    torch.mean((output - batch) ** 2, dim=-1).cpu().numpy()
                )
        return overlap_average(np.concatenate(rows, axis=0), len(test))

    scores, timing = measure_inference(infer)
    save_standard_outputs(
        output_dir=args.output_dir,
        method="PUAD",
        dataset=args.dataset,
        seed=args.seed,
        scores=scores,
        labels=labels,
        timing=timing,
        implementation={
            "upstream": "https://github.com/LiYuxin321/PUAD.git",
            "commit": "41e8b4377e6baa83e56b8f3acdb60ff04ed6c892",
            "adapter": "official PUAD architecture with generic single-series loader",
            "protocol_disclosure": (
                "The upstream runner only implements SMD/MSL multi-task file layouts "
                "and hard-codes its seed. The adapter preserves the published "
                "architecture/loss terms but uses each benchmark's normal prefix as "
                "one training task so all five datasets share one data protocol."
            ),
        },
    )


if __name__ == "__main__":
    main()
