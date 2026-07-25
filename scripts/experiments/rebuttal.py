from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.experiments.manifest import RunSpec, artifact_is_complete, artifact_is_reusable
    from scripts.experiments.protocol import (
        build_lite_tasks,
        build_p0_tasks,
        build_p1_tasks,
    )
    from scripts.experiments.runner import (
        execute_tasks,
        execution_is_reusable,
        freeze_environment,
        write_plan,
    )
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    from .manifest import RunSpec, artifact_is_complete, artifact_is_reusable
    from .protocol import build_lite_tasks, build_p0_tasks, build_p1_tasks
    from .runner import execute_tasks, execution_is_reusable, freeze_environment, write_plan


AC_CORE_GROUPS = {
    "anchors",
    "baselines",
    "catch",
    "efficiency",
    "e9",
    "mechanism",
    "tep",
}


def select_tasks(
    *,
    scope: str,
    groups: set[str] | None,
    artifact_root: Path,
    python_exe: str,
    lite_anchor_root: Path | None = None,
    methods: set[str] | None = None,
    datasets: set[str] | None = None,
) -> list[RunSpec]:
    if scope not in {"ac-core", "lite", "p0", "p1", "all"}:
        raise ValueError("scope must be one of ac-core, lite, p0, p1, all")
    tasks: list[RunSpec] = []
    if scope == "lite":
        tasks.extend(
            build_lite_tasks(
                artifact_root,
                python_exe=python_exe,
                anchor_root=lite_anchor_root,
            )
        )
    elif lite_anchor_root is not None:
        raise ValueError("--lite-anchor-root is only valid with --scope lite.")
    if scope in {"ac-core", "p0", "all"}:
        tasks.extend(build_p0_tasks(artifact_root, python_exe=python_exe))
    if scope in {"p1", "all"}:
        tasks.extend(build_p1_tasks(artifact_root, python_exe=python_exe))
    if scope == "ac-core":
        tasks = [task for task in tasks if task.group in AC_CORE_GROUPS]
        # The running A/B shard contract is frozen at 102 tasks. Native
        # KNN/LOF were added later to the independent lightweight queue.
        tasks = [task for task in tasks if task.method not in {"KNN", "LOF"}]
    if groups:
        tasks = [task for task in tasks if task.group in groups]
    if methods:
        tasks = [task for task in tasks if task.method in methods]
    if datasets:
        tasks = [task for task in tasks if task.dataset in datasets]
    if len({task.run_id for task in tasks}) != len(tasks):
        raise ValueError("Selected task set contains duplicate run IDs.")
    return tasks


def _apply_python_map(tasks: Iterable[RunSpec], path: Path | None) -> list[RunSpec]:
    if path is None:
        return list(tasks)
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("--python-map must contain a JSON object.")
    rewritten: list[RunSpec] = []
    for task in tasks:
        executable = mapping.get(task.method)
        if not executable or not task.command or task.command[0] == "bash":
            rewritten.append(task)
            continue
        command = (str(executable), *task.command[1:])
        rewritten.append(
            RunSpec(
                method=task.method,
                dataset=task.dataset,
                seed=task.seed,
                stage=task.stage,
                group=task.group,
                compute_kind=task.compute_kind,
                config=task.config,
                command=command,
                artifact_dir=task.artifact_dir,
                required_artifacts=task.required_artifacts,
                dependencies=task.dependencies,
                metadata=task.metadata,
            )
        )
    return rewritten


def _replace_option(command: tuple[str, ...], names: set[str], value: str) -> tuple[str, ...]:
    rewritten = list(command)
    for index, token in enumerate(rewritten[:-1]):
        if token in names:
            rewritten[index + 1] = value
    return tuple(rewritten)


def _data_root_key(task: RunSpec) -> tuple[str, ...]:
    edition = task.config.get("edition")
    if edition:
        return (f"TSB-AD-{edition}", "TSB-AD", "default")
    if task.dataset == "TEP":
        return ("TEP", "default")
    return (task.dataset, "anomaly_detect", "default")


def _apply_data_root_map(tasks: Iterable[RunSpec], path: Path | None) -> list[RunSpec]:
    rows = list(tasks)
    if path is None:
        return rows
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("--data-root-map must contain a JSON object.")

    first_pass: list[tuple[RunSpec, str]] = []
    for task in rows:
        selected = next(
            (mapping[key] for key in _data_root_key(task) if key in mapping),
            None,
        )
        if selected is None:
            first_pass.append((task, task.run_id))
            continue
        data_root = str(Path(str(selected)).expanduser())
        command = _replace_option(task.command, {"--data_root", "--data-root"}, data_root)
        metadata = dict(task.metadata)
        if task.dataset == "TEP":
            command = (*command, data_root)
            metadata["data_paths"] = [data_root]
        elif task.dataset == "ALL" and any(
            option in task.command for option in ("--data_root", "--data-root")
        ):
            metadata["data_paths"] = [data_root]
        config = {**task.config, "data_root_override": data_root}
        first_pass.append(
            (
                RunSpec(
                    method=task.method,
                    dataset=task.dataset,
                    seed=task.seed,
                    stage=task.stage,
                    group=task.group,
                    compute_kind=task.compute_kind,
                    config=config,
                    command=command,
                    artifact_dir=task.artifact_dir,
                    required_artifacts=task.required_artifacts,
                    dependencies=task.dependencies,
                    metadata=metadata,
                ),
                task.run_id,
            )
        )

    remap = {old_id: task.run_id for task, old_id in first_pass}
    return [
        RunSpec(
            method=task.method,
            dataset=task.dataset,
            seed=task.seed,
            stage=task.stage,
            group=task.group,
            compute_kind=task.compute_kind,
            config=task.config,
            command=task.command,
            artifact_dir=task.artifact_dir,
            required_artifacts=task.required_artifacts,
            dependencies=tuple(remap.get(dep, dep) for dep in task.dependencies),
            metadata=task.metadata,
        )
        for task, _ in first_pass
    ]


def status_payload(tasks: Iterable[RunSpec], *, repo_root: Path = REPO_ROOT) -> dict:
    counts: Counter[str] = Counter()
    rows = []
    for task in tasks:
        record_path = task.artifact_dir / "run_record.json"
        if execution_is_reusable(task, repo_root):
            status = "completed"
        elif artifact_is_complete(task):
            status = "unverified_artifacts"
        elif record_path.is_file():
            try:
                status = str(
                    json.loads(record_path.read_text(encoding="utf-8")).get(
                        "status", "unknown"
                    )
                )
            except (OSError, json.JSONDecodeError):
                status = "invalid_record"
        else:
            status = "pending"
        counts[status] += 1
        rows.append({"run_id": task.run_id, "group": task.group, "status": status})
    return {"counts": dict(sorted(counts.items())), "rows": rows}


def collect_records(
    tasks: Iterable[RunSpec],
    output_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict:
    rows = []
    for task in tasks:
        record_path = task.artifact_dir / "run_record.json"
        record = (
            json.loads(record_path.read_text(encoding="utf-8"))
            if record_path.is_file()
            else {}
        )
        rows.append(
            {
                "run_id": task.run_id,
                "method": task.method,
                "dataset": task.dataset,
                "group": task.group,
                "compute_kind": task.compute_kind,
                "status": (
                    record.get("status", "pending")
                    if execution_is_reusable(task, repo_root)
                    else (
                        "unverified_artifacts"
                        if artifact_is_complete(task)
                        else record.get("status", "pending")
                    )
                ),
                "runtime_seconds": record.get("runtime_seconds"),
                "artifact_dir": str(task.artifact_dir),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "records.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {"total": len(rows), "status": dict(Counter(row["status"] for row in rows))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan and execute formal SCAR rebuttal experiments.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "run", "resume", "status", "validate", "collect"):
        command = subparsers.add_parser(action)
        command.add_argument(
            "--scope",
            choices=["ac-core", "lite", "p0", "p1", "all"],
            default="all",
        )
        command.add_argument("--group", nargs="+", default=None)
        command.add_argument("--method", nargs="+", default=None)
        command.add_argument("--dataset", nargs="+", default=None)
        command.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts")
        command.add_argument(
            "--manifest-dir",
            type=Path,
            default=REPO_ROOT / "artifacts" / "rebuttal_manifest",
        )
        command.add_argument("--python", default=sys.executable)
        command.add_argument("--python-map", type=Path, default=None)
        command.add_argument("--data-root-map", type=Path, default=None)
        command.add_argument("--lite-anchor-root", type=Path, default=None)
        command.add_argument("--max-parallel", type=int, default=1)
        command.add_argument("--gpu-devices", nargs="+", default=None)
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--failed-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = select_tasks(
        scope=args.scope,
        groups=set(args.group) if args.group else None,
        artifact_root=args.artifact_root.resolve(),
        python_exe=str(args.python),
        lite_anchor_root=(
            args.lite_anchor_root.resolve()
            if args.lite_anchor_root is not None
            else None
        ),
        methods=set(args.method) if args.method else None,
        datasets=set(args.dataset) if args.dataset else None,
    )
    tasks = _apply_python_map(tasks, args.python_map)
    tasks = _apply_data_root_map(tasks, args.data_root_map)
    if args.action == "plan":
        payload = write_plan(tasks, args.manifest_dir.resolve())
        freeze_environment(
            args.manifest_dir.resolve() / "environment",
            repo_root=REPO_ROOT,
        )
        print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
        return 0
    if args.action == "status":
        payload = status_payload(tasks, repo_root=REPO_ROOT)
        print(json.dumps(payload["counts"], ensure_ascii=False, sort_keys=True))
        return 0
    if args.action == "validate":
        incomplete = [
            task for task in tasks if not execution_is_reusable(task, REPO_ROOT)
        ]
        print(json.dumps({"total": len(tasks), "incomplete": len(incomplete)}))
        return int(bool(incomplete))
    if args.action == "collect":
        payload = collect_records(
            tasks,
            args.manifest_dir.resolve(),
            repo_root=REPO_ROOT,
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    freeze_environment(
        args.manifest_dir.resolve() / "environment",
        repo_root=REPO_ROOT,
    )
    results = execute_tasks(
        tasks,
        repo_root=REPO_ROOT,
        max_parallel=args.max_parallel,
        gpu_devices=args.gpu_devices,
        resume=args.action == "resume",
        failed_only=args.failed_only,
        dry_run=args.dry_run,
    )
    counts = Counter(result["status"] for result in results)
    print(json.dumps(dict(counts), ensure_ascii=False, sort_keys=True))
    return int(any(status == "failed" for status in counts))


if __name__ == "__main__":
    raise SystemExit(main())
