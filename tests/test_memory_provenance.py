from __future__ import annotations

from pathlib import Path

import torch

from coremad.config import CoReMADConfig
from coremad.memory import MemoryBank, ScaleMemory


def make_config(tmp_path: Path, **overrides) -> CoReMADConfig:
    values = {
        "artifact_root": str(tmp_path),
        "experiment_name": "memory_provenance",
        "device": "cpu",
        "n_channels": 2,
        "seq_len": 8,
        "patch_sizes": [2],
        "d_state": 2,
        "d_z": 2,
        "state_prototype_count": 1,
        "use_prototype_support": False,
        "use_faiss": False,
        "faiss_use_gpu": False,
        "top_M": 2,
        "top_K": 2,
        "state_novelty_k": 1,
    }
    values.update(overrides)
    return CoReMADConfig(**values)


def test_memory_seed_defaults_to_stage_a_seed(tmp_path: Path) -> None:
    config = make_config(tmp_path, seed=17)
    assert config.effective_memory_seed == 17

    independent = make_config(tmp_path, seed=17, memory_seed=91)
    assert independent.effective_memory_seed == 91


def test_scale_memory_rejects_misaligned_raw_starts() -> None:
    try:
        ScaleMemory(
            z=torch.zeros(2, 2),
            c=torch.zeros(2, 2),
            window_ids=torch.tensor([0, 1]),
            raw_starts=torch.tensor([4]),
        )
    except ValueError as exc:
        assert "raw_starts" in str(exc)
    else:
        raise AssertionError("ScaleMemory accepted misaligned raw_starts")


def test_memory_save_load_preserves_raw_starts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    scale = ScaleMemory(
        z=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        c=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        window_ids=torch.tensor([0, 1]),
        raw_starts=torch.tensor([3, 11]),
    )
    memory = MemoryBank(
        config=config,
        state_bank=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        state_window_starts=torch.tensor([3, 9]),
        scales=[scale],
    )
    memory.save(config.memory_path)

    loaded = MemoryBank.load(config.memory_path, config)
    assert loaded.scales[0].raw_starts.tolist() == [3, 11]


def test_old_memory_payload_marks_unknown_raw_starts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    payload = {
        "state_bank": torch.tensor([[0.0, 0.0]]),
        "state_window_starts": torch.tensor([4]),
        "prototypes": {
            "centers": torch.zeros(0, 2),
            "labels": torch.zeros(0, dtype=torch.long),
            "members": [],
        },
        "scales": [
            {
                "z": torch.tensor([[0.0, 0.0]]),
                "c": torch.tensor([[0.0, 0.0]]),
                "window_ids": torch.tensor([0]),
                "coverage_threshold": float("inf"),
            }
        ],
    }
    torch.save(payload, config.memory_path)

    loaded = MemoryBank.load(config.memory_path, config)
    assert loaded.scales[0].raw_starts.tolist() == [-1]


def test_query_details_include_window_and_patch_starts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    scale = ScaleMemory(
        z=torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        c=torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        window_ids=torch.tensor([0, 1]),
        raw_starts=torch.tensor([5, 17]),
    )
    memory = MemoryBank(
        config=config,
        state_bank=torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        state_window_starts=torch.tensor([5, 15]),
        scales=[scale],
    )

    details = memory.query_preencoded(
        state_vec=torch.tensor([[0.1, 0.1]]),
        z_list=[torch.tensor([[[0.1, 0.1]]])],
        c_list=[torch.tensor([[[0.1, 0.1]]])],
        return_details=True,
    )

    assert details["coarse_window_starts"].shape == (1, 2)
    assert details["neighbor_raw_starts"][0].shape == (1, 1, 2)
    assert details["neighbor_raw_starts"][0][0, 0, 0].item() == 5


def test_full_memory_audit_records_patch_decisions(tmp_path: Path) -> None:
    config = make_config(tmp_path, memory_audit_mode="full")
    MemoryBank._write_array_audit(
        patch_size=2,
        raw_starts=torch.tensor([0, 2, 4]),
        window_ids=torch.tensor([0, 0, 1]),
        completion_scores=torch.tensor([0.1, 0.5, 0.2]),
        clean_mask=torch.tensor([True, False, True]),
        raw_keep_indices=torch.tensor([2]),
        clean_threshold=0.2,
        config=config,
    )

    import numpy as np

    with np.load(config.memory_audit_dir / "memory_audit_scale2.npz") as payload:
        assert payload["raw_start"].tolist() == [0, 2, 4]
        assert payload["kept_after_purification"].tolist() == [True, False, True]
        assert payload["kept_after_coreset"].tolist() == [False, False, True]
