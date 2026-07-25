from __future__ import annotations

import torch

from coremad.memory import MemoryBank


def test_batched_prototype_support_matches_per_sample_reference() -> None:
    generator = torch.Generator().manual_seed(42)
    batch, queries, candidates, width, top_k = 3, 4, 7, 5, 3
    retrieval_query = torch.randn(batch, queries, width, generator=generator)
    query_z = torch.randn(batch, queries, width, generator=generator)
    candidate_z = torch.randn(batch, candidates, width, generator=generator)
    candidate_c = torch.randn(batch, candidates, width, generator=generator)
    candidate_window_ids = torch.arange(candidates).repeat(batch, 1)
    candidate_valid = torch.zeros(batch, candidates, dtype=torch.bool)
    candidate_valid[0, :7] = True
    candidate_valid[1, :5] = True
    candidate_valid[2, :2] = True
    candidate_weights = torch.rand(batch, candidates, generator=generator) + 0.1

    actual = MemoryBank._batched_prototype_support_score(
        retrieval_query,
        query_z,
        candidate_z,
        candidate_c,
        candidate_window_ids,
        candidate_valid,
        candidate_weights,
        use_context_key_retrieval=True,
        top_k=top_k,
        tau=0.7,
        eps=1e-8,
    )

    expected_rows = []
    for row, valid_count in enumerate((7, 5, 2)):
        neighbor_z, _, neighbor_valid, _, topk_idx = (
            MemoryBank._context_topk_neighbors(
                retrieval_query[row],
                candidate_c[row, :valid_count],
                candidate_z[row, :valid_count],
                candidate_window_ids[row, :valid_count],
                top_k,
            )
        )
        neighbor_weights = candidate_weights[row, :valid_count][topk_idx]
        expected_rows.append(
            MemoryBank._compute_support_score(
                query_z[row],
                neighbor_z,
                neighbor_valid,
                neighbor_weights,
                tau=0.7,
                eps=1e-8,
            )
        )
    expected = torch.stack(expected_rows)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)
