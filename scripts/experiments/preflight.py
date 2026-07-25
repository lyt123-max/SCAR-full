from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.protocol import build_p0_tasks, build_p1_tasks


PINNED = {
    "PaAno": "d4c67116190efa4592dc6a8a157ced0def68b6af",
    "PUAD": "41e8b4377e6baa83e56b8f3acdb60ff04ed6c892",
    "PGRF-Net": "5dc6f7522d20043eb31f6b2b13091c80ad394dcb",
    "CATCH": "3647c69be5eb56649b072596cf89098e689e20c3",
}


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", *args],
        cwd=path,
        text=True,
        encoding="utf-8",
        errors="replace",
        stderr=subprocess.STDOUT,
    ).strip()


def static_checks(
    artifact_root: Path,
    *,
    require_upstreams: bool = True,
) -> list[dict]:
    checks = []

    def add(name: str, ok: bool, detail: object) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    p0 = build_p0_tasks(artifact_root, python_exe=sys.executable)
    p1 = build_p1_tasks(artifact_root, python_exe=sys.executable)
    add("p0_unique_run_ids", len({task.run_id for task in p0}) == len(p0), len(p0))
    add(
        "p0_compute_counts",
        sum(task.compute_kind == "full_model_fit" for task in p0) == 659
        and sum(task.compute_kind == "stage_b_test" for task in p0) == 180,
        {
            "full": sum(task.compute_kind == "full_model_fit" for task in p0),
            "stage_b_test": sum(task.compute_kind == "stage_b_test" for task in p0),
        },
    )
    tsb = [task for task in p0 if task.group == "tsb"]
    add("tsb_official_count", len(tsb) == 598, len(tsb))
    add(
        "formal_seed_42",
        all(task.seed == 42 for task in (*p0, *p1)),
        sorted({task.seed for task in (*p0, *p1)}),
    )
    add(
        "formal_forbids_eval_full",
        not any(
            str(task.config.get("edition", "")).upper() == "U"
            and str(task.config.get("split", "")).lower().replace("-", "_")
            == "eval_full"
            for task in (*p0, *p1)
        ),
        None,
    )
    add(
        "formal_forbids_plot_commands",
        not any("plot" in Path(part).name.lower() for task in (*p0, *p1) for part in task.command),
        None,
    )
    baseline = [task for task in p0 if task.group == "baselines"]
    add(
        "baseline_matrix_5x5_without_memto_or_catch_reruns",
        len(baseline) == 25
        and {task.method for task in baseline}
        == {"PaAno", "PUAD", "PGRF-Net", "KNN", "LOF"},
        {"count": len(baseline), "methods": sorted({task.method for task in baseline})},
    )
    e10_protocol = [task for task in p0 if task.group == "e10_protocol"]
    e10_runs = [task for task in p0 if task.group == "e10"]
    e10_zero = [task for task in p0 if task.group == "e10_zero"]
    add(
        "e10_frozen_protocol",
        len(e10_protocol) == 5
        and len(e10_runs) == 135
        and len(e10_zero) == 30
        and all("--fold_manifest" in task.command for task in e10_runs)
        and all("--fold-manifest" in task.command for task in e10_zero),
        {
            "protocols": len(e10_protocol),
            "nonzero": len(e10_runs),
            "zero": len(e10_zero),
        },
    )
    tep = [task for task in p0 if task.group == "tep"]
    add(
        "tep_sequence_outputs",
        len(tep) == 1
        and "test_sequence_scores_selected.npy" in tep[0].required_artifacts
        and "test_scores_selected.npy" not in tep[0].required_artifacts,
        tep[0].required_artifacts if tep else None,
    )
    add("p1_unique_run_ids", len({task.run_id for task in p1}) == len(p1), len(p1))

    if require_upstreams:
        for method, expected in PINNED.items():
            path = REPO_ROOT / "third_party" / "baselines" / method
            if not path.is_dir():
                add(f"upstream_{method}", False, f"missing {path}")
                continue
            actual = _git(path, "rev-parse", "HEAD")
            dirty = _git(path, "status", "--porcelain")
            add(
                f"upstream_{method}",
                actual == expected and not dirty,
                {"commit": actual, "dirty": bool(dirty)},
            )
    expected_envs = (
        "scar-paano-cu126.yml",
        "catch-cu126.yml",
        "pgrf-cu126.yml",
        "puad-official.yml",
        "puad-cu126-compat.yml",
    )
    add(
        "environment_specs",
        all((REPO_ROOT / "environments" / name).is_file() for name in expected_envs),
        list(expected_envs),
    )
    return checks


def runtime_checks() -> list[dict]:
    checks = []
    try:
        import numpy as np

        checks.append(
            {
                "name": "numpy_version",
                "ok": np.__version__ == "1.26.4",
                "detail": np.__version__,
            }
        )
    except ImportError as exc:
        checks.append({"name": "numpy_runtime", "ok": False, "detail": str(exc)})
    try:
        import torch

        checks.append(
            {
                "name": "torch_version",
                "ok": torch.__version__.startswith("2.7.1"),
                "detail": torch.__version__,
            }
        )
        checks.append(
            {
                "name": "cuda_runtime",
                "ok": str(torch.version.cuda) == "12.6",
                "detail": torch.version.cuda,
            }
        )
        checks.append(
            {
                "name": "cuda_available",
                "ok": torch.cuda.is_available(),
                "detail": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except ImportError as exc:
        checks.append({"name": "torch_runtime", "ok": False, "detail": str(exc)})
    try:
        from TSB_AD.evaluation.metrics import get_metrics

        checks.append(
            {
                "name": "tsb_ad_metrics",
                "ok": callable(get_metrics),
                "detail": "TSB_AD.evaluation.metrics.get_metrics",
            }
        )
    except ImportError as exc:
        checks.append({"name": "tsb_ad_metrics", "ok": False, "detail": str(exc)})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight formal rebuttal execution.")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-runtime", action="store_true")
    args = parser.parse_args()
    checks = static_checks(args.artifact_root.resolve())
    if args.strict_runtime:
        checks.extend(runtime_checks())
    payload = {
        "schema_version": 1,
        "strict_runtime": args.strict_runtime,
        "passed": all(item["ok"] for item in checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": payload["passed"], "checks": len(checks)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
