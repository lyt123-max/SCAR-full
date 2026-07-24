from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate active Markdown experiment protocol.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = (
        REPO_ROOT / "rebuttal实验运行清单.md",
        REPO_ROOT / "rebuttal执行计划.md",
        REPO_ROOT / "PROJECT_PLAN.md",
    )
    texts = {path.name: path.read_text(encoding="utf-8") for path in paths}
    joined = "\n".join(texts.values())
    checks = {
        "e9_contains_0p005": "0.005" in joined,
        "stronger_is_0p10": "stronger (`0.10`)" in joined
        or "stronger purification `clean_ratio=0.10`" in joined,
        "no_stale_stronger_0p05": "stronger (`0.05`)" not in joined
        and "stronger purification `clean_ratio=0.05`" not in joined,
        "p0_counts": "659 次 full model fit" in joined
        and "180 次 Stage-B/Test" in joined,
        "tsb_official_598": "598" in joined and "U-Eva 350" in joined,
        "formal_table_only": "不提交任何图" in joined or "不允许提交图片" in joined,
        "central_cli_documented": "scripts/experiments/rebuttal.py" in joined,
        "e10_frozen_protocol": "contamination_fold_manifest.json" in joined,
        "data_root_map_documented": "--data-root-map" in joined,
        "p0_table_collector": "p0_tables" in joined,
        "manifest_total_1230": "1230" in joined,
    }
    payload = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
