from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from .common import (
        REPO_ROOT,
        measure_inference,
        save_standard_outputs,
        set_seed,
    )
except ImportError:
    from common import (
        REPO_ROOT,
        measure_inference,
        save_standard_outputs,
        set_seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCAR-side PaAno adapter.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Formal PaAno runs require seed 42.")
    official = REPO_ROOT / "third_party" / "baselines" / "PaAno"
    sys.path.insert(0, str(official))
    import torch
    import main as paano

    set_seed(args.seed)
    files = sorted((args.data_dir / "paano").glob("*.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one PaAno CSV, found {len(files)}.")
    train, train_labels, test, test_labels = paano.load_and_split_data(str(files[0]))
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int8)
    test_labels = np.asarray(test_labels, dtype=np.int8)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1e-8
    train = (train - mean) / std
    test = (test - mean) / std
    full = np.concatenate([train, test], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    creator = paano.PatchCreator(L=args.patch_size, s=1, random_seed=args.seed)
    train_loader, test_loader, _ = creator.create_dataloaders(
        train, full, labels, batch_size=args.batch_size
    )
    batch, _ = next(iter(train_loader))
    model = paano.PatchEncoder(in_channels=batch.shape[1], use_revin=True).to(device)
    paano.train_model(
        model,
        train_loader,
        paano.preprocess_to_patches(train, patch_size=args.patch_size, stride=1),
        device,
        num_iter=args.num_iters,
        pretext_step=args.patch_size,
        lr=1e-4,
        see_loss=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    memory_bank, _ = paano.create_memory_bank(
        model, train_loader, device, num_cores=0.1
    )

    def infer() -> np.ndarray:
        patch_scores = paano.calculate_anomaly_scores(
            model, test_loader, memory_bank, top_k=3, device=device
        )
        return paano.distribute_patch_scores_to_points(
            patch_scores,
            patch_size=args.patch_size,
            num_points=len(labels),
        )

    scores, timing = measure_inference(infer)
    save_standard_outputs(
        output_dir=args.output_dir,
        method="PaAno",
        dataset=args.dataset,
        seed=args.seed,
        scores=scores[len(train) :],
        labels=test_labels,
        timing=timing,
        implementation={
            "upstream": "https://github.com/jinnnju/PaAno.git",
            "commit": "d4c67116190efa4592dc6a8a157ced0def68b6af",
            "adapter": (
                "official run_mul hyperparameters (patch 96, 100 iterations, RevIN); "
                "formal seed 42; test-only aligned export"
            ),
        },
    )


if __name__ == "__main__":
    main()
