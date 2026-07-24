from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Single-GPU SCAR-side MEMTO adapter.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _windows(torch, values: np.ndarray, width: int, step: int):
    if len(values) < width:
        raise ValueError(f"Series length {len(values)} is below MEMTO window {width}.")
    starts = range(0, len(values) - width + 1, step)
    return torch.from_numpy(np.stack([values[start : start + width] for start in starts]))


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Formal MEMTO runs require seed 42.")
    official = REPO_ROOT / "third_party" / "baselines" / "MEMTO"
    sys.path.insert(0, str(official))
    import torch
    import torch.nn as nn
    from kmeans_pytorch import kmeans
    from model.Transformer import TransformerVar
    from model.loss_functions import EntropyLoss, GatheringLoss

    set_seed(args.seed)
    train, test, labels = load_common_data(args.data_dir)
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train).astype(np.float32)
    test = scaler.transform(test).astype(np.float32)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_channels = train.shape[1]
    d_model = 512
    n_memory = 128

    train_windows = _windows(torch, train, args.window_size, args.window_size)
    loader = torch.utils.data.DataLoader(
        train_windows, batch_size=args.batch_size, shuffle=True
    )
    entropy_loss = EntropyLoss()
    reconstruction = nn.MSELoss()

    def make_model(memory=None):
        model = TransformerVar(
            win_size=args.window_size,
            enc_in=input_channels,
            c_out=input_channels,
            n_memory=n_memory,
            d_model=d_model,
            e_layers=3,
            device=device,
            memory_init_embedding=memory,
            memory_initial=False,
            phase_type=None,
            dataset_name=args.dataset,
        )
        return model.to(device)

    def fit(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()
        for _ in range(args.epochs):
            for batch in loader:
                batch = batch.float().to(device)
                payload = model(batch)
                loss = reconstruction(payload["out"], batch)
                loss += 0.01 * entropy_loss(payload["attn"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    first_model = make_model()
    fit(first_model)
    first_model.eval()
    query_rows = []
    with torch.inference_mode():
        for batch in loader:
            query_rows.append(first_model(batch.float().to(device))["queries"])
    query_values = torch.cat(query_rows, dim=0).reshape(-1, d_model)
    _, centers = kmeans(
        X=query_values,
        num_clusters=n_memory,
        distance="euclidean",
        device=device,
    )
    model = make_model(centers.detach())
    fit(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")

    gathering = GatheringLoss(reduce=False)
    criterion = nn.MSELoss(reduction="none")

    def infer() -> np.ndarray:
        windows = _windows(torch, test, args.window_size, 1)
        batches = torch.utils.data.DataLoader(
            windows, batch_size=max(args.batch_size, 64), shuffle=False
        )
        rows = []
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                batch = batch.float().to(device)
                payload = model(batch)
                rec = torch.mean(criterion(batch, payload["out"]), dim=-1)
                latent = torch.softmax(
                    gathering(payload["queries"], payload["mem"]) / 0.1, dim=-1
                )
                rows.append((latent * rec).detach().cpu().numpy())
        return overlap_average(np.concatenate(rows, axis=0), len(test))

    scores, timing = measure_inference(infer)
    save_standard_outputs(
        output_dir=args.output_dir,
        method="MEMTO",
        dataset=args.dataset,
        seed=args.seed,
        scores=scores,
        labels=labels,
        timing=timing,
        implementation={
            "upstream": "https://github.com/gunny97/MEMTO.git",
            "commit": "5a3287103021c5c7e7cac9377c626cf18bdea50c",
            "adapter": (
                "official model/losses; project-side single-GPU training and "
                "overlap-averaged raw-score export"
            ),
            "compatibility_changes": [
                "replace upstream four-GPU DataParallel with selected single device",
                "export raw point scores omitted by upstream test()",
            ],
        },
    )


if __name__ == "__main__":
    main()
