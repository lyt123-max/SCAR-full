from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .manifest import RunSpec, artifact_is_complete, artifact_is_reusable, canonical_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=256)
def hash_path(path: Path) -> str:
    path = path.resolve()
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
        return digest.hexdigest()
    if not path.is_dir():
        return ""
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(str(child.stat().st_size).encode("ascii"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def _untracked_source_files(repo_root: Path) -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    source_roots = {"coremad", "scripts", "tests", "environments", "docs"}
    root_suffixes = {".py", ".md", ".json", ".yml", ".yaml", ".toml"}
    selected = []
    for line in output.splitlines():
        relative = Path(line.strip())
        if not relative.parts:
            continue
        if relative.parts[0] in source_roots or (
            len(relative.parts) == 1 and relative.suffix.lower() in root_suffixes
        ):
            path = repo_root / relative
            if path.is_file():
                selected.append(path)
    return sorted(selected)


@lru_cache(maxsize=8)
def source_fingerprint(repo_root_text: str) -> str:
    repo_root = Path(repo_root_text)
    digest = hashlib.sha256()
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "diff", "--binary", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        digest.update(commit)
        digest.update(diff)
        for path in _untracked_source_files(repo_root):
            digest.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
            digest.update(sha256_file(path).encode("ascii"))
    except (OSError, subprocess.CalledProcessError):
        digest.update(b"unavailable")
    return digest.hexdigest()


def _command_value(command: tuple[str, ...], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _data_paths(task: RunSpec, repo_root: Path) -> list[Path]:
    declared = task.metadata.get("data_paths", [])
    if declared:
        return [Path(path) for path in declared]
    data_root_value = _command_value(task.command, "--data_root") or _command_value(
        task.command, "--data-root"
    )
    if data_root_value is not None:
        data_root = Path(data_root_value)
        if not data_root.is_absolute():
            data_root = repo_root / data_root
        if task.dataset == "ALL" and data_root.is_dir():
            return [data_root]
        direct = data_root / task.dataset
        candidates = [
            direct,
            direct.with_suffix(".csv"),
            data_root / "data" / f"{task.dataset}.csv",
        ]
        matched = next((path for path in candidates if path.is_file()), None)
        if matched is not None:
            return [matched]
    catch_file = repo_root / "dataset" / "anomaly_detect" / "data" / f"{task.dataset}.csv"
    if catch_file.is_file():
        return [catch_file]
    source = task.config.get("source_experiment") or task.config.get("base_experiment_dir")
    if source:
        config_path = Path(str(source)) / "config.json"
        if config_path.is_file():
            return [config_path]
    if task.dataset == "TEP":
        tep_root = repo_root / "Multi-mode-Fault-Diagnosis-Datasets-with-TE-process"
        if tep_root.is_dir():
            return [tep_root]
    return []


def data_fingerprint(task: RunSpec, repo_root: Path) -> tuple[str | None, list[str]]:
    paths = [path.resolve() for path in _data_paths(task, repo_root)]
    if not paths:
        return None, []
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(hash_path(path).encode("ascii"))
    return digest.hexdigest(), [str(path) for path in paths]


def execution_is_reusable(task: RunSpec, repo_root: Path) -> bool:
    if not artifact_is_reusable(task):
        return False
    try:
        record = json.loads(
            (task.artifact_dir / "run_record.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    current_data_hash, _ = data_fingerprint(task, repo_root)
    return (
        record.get("source_hash") == source_fingerprint(str(repo_root.resolve()))
        and record.get("data_hash") == current_data_hash
        and record.get("config_hash")
        == hashlib.sha256(canonical_json(task.config).encode("utf-8")).hexdigest()
    )


def environment_snapshot(repo_root: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cwd": str(repo_root.resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }
    try:
        snapshot["git_commit"] = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root.resolve()}", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        ).strip()
        snapshot["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "-c", f"safe.directory={repo_root.resolve()}", "status", "--porcelain"],
                cwd=repo_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        snapshot["git_commit"] = None
        snapshot["git_dirty"] = None
    return snapshot


def _capture_command(command: list[str], *, cwd: Path) -> str:
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        prefix = f"$ {subprocess.list2cmdline(command)}\n[exit_code] {process.returncode}\n"
        return prefix + process.stdout
    except OSError as exc:
        return (
            f"$ {subprocess.list2cmdline(command)}\n[unavailable] "
            f"{type(exc).__name__}: {exc}\n"
        )


def freeze_environment(
    output_dir: Path,
    *,
    repo_root: Path,
    include_package_commands: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = environment_snapshot(repo_root)
    packages = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "faiss-cpu", "faiss-gpu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    snapshot["packages"] = packages
    (output_dir / "environment.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    git_diff = _capture_command(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.resolve()}",
            "diff",
            "--binary",
            "HEAD",
        ],
        cwd=repo_root,
    )
    (output_dir / "git_diff.patch").write_text(git_diff, encoding="utf-8")
    untracked_sources = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in _untracked_source_files(repo_root)
    ]
    (output_dir / "untracked_source_files.json").write_text(
        json.dumps(untracked_sources, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if include_package_commands:
        pip_freeze = _capture_command(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            cwd=repo_root,
        )
        conda_explicit = _capture_command(["conda", "list", "--explicit"], cwd=repo_root)
    else:
        pip_freeze = "[not captured in lightweight test]\n"
        conda_explicit = "[not captured in lightweight test]\n"
    (output_dir / "pip_freeze.txt").write_text(pip_freeze, encoding="utf-8")
    (output_dir / "conda_explicit.txt").write_text(conda_explicit, encoding="utf-8")
    (output_dir / "nvidia_smi.txt").write_text(
        _capture_command(["nvidia-smi", "-q"], cwd=repo_root),
        encoding="utf-8",
    )
    return snapshot


def summarize_tasks(tasks: Iterable[RunSpec]) -> dict[str, int]:
    rows = list(tasks)
    summary: dict[str, int] = {"total": len(rows)}
    for task in rows:
        summary[task.compute_kind] = summary.get(task.compute_kind, 0) + 1
    for task in rows:
        key = f"group:{task.group}"
        summary[key] = summary.get(key, 0) + 1
    return summary


def write_plan(tasks: Iterable[RunSpec], output_dir: Path) -> dict[str, Any]:
    rows = list(tasks)
    if len({task.run_id for task in rows}) != len(rows):
        raise ValueError("Experiment plan contains duplicate run_id values.")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "summary": summarize_tasks(rows),
        "tasks": [task.as_dict() for task in rows],
    }
    (output_dir / "plan.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "plan.jsonl").write_text(
        "".join(json.dumps(task.as_dict(), ensure_ascii=False) + "\n" for task in rows),
        encoding="utf-8",
    )
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def execute_task(
    task: RunSpec,
    *,
    repo_root: Path,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if resume and execution_is_reusable(task, repo_root):
        return {**task.as_dict(), "status": "skipped_complete", "artifacts_complete": True}
    if dry_run:
        return {**task.as_dict(), "status": "planned", "artifacts_complete": False}

    task.artifact_dir.mkdir(parents=True, exist_ok=True)
    record_path = task.artifact_dir / "run_record.json"
    log_path = task.artifact_dir / f"{task.stage}.log"
    runtime_command = task.command
    if resume and "--resume" not in runtime_command:
        if any(Path(part).name == "run.py" for part in runtime_command):
            runtime_command = (*runtime_command, "--resume", "1")
    runtime_environment = os.environ.copy()
    for key, value in task.metadata.get("environment", {}).items():
        runtime_environment[str(key)] = str(value)
    if resume:
        runtime_environment["RESUME"] = "1"
    record = {
        **task.as_dict(),
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "cwd": str(repo_root.resolve()),
        "config_hash": hashlib.sha256(canonical_json(task.config).encode("utf-8")).hexdigest(),
        "source_hash": source_fingerprint(str(repo_root.resolve())),
        "environment": environment_snapshot(repo_root),
        "log": str(log_path.resolve()),
        "executed_command": list(runtime_command),
        "resume_requested": bool(resume),
        "environment_overrides": task.metadata.get("environment", {}),
    }
    record["data_hash"], record["data_paths"] = data_fingerprint(task, repo_root)
    _atomic_json(record_path, record)
    started = time.perf_counter()
    try:
        with log_path.open("a", encoding="utf-8", newline="") as log:
            log.write("$ " + subprocess.list2cmdline(list(runtime_command)) + "\n")
            process = subprocess.run(
                list(runtime_command),
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=runtime_environment,
                check=False,
            )
        return_code = int(process.returncode)
        error = None
    except Exception as exc:
        return_code = -1
        error = f"{type(exc).__name__}: {exc}"
    complete = return_code == 0 and artifact_is_complete(task)
    record.update(
        {
            "status": "completed" if complete else "failed",
            "return_code": return_code,
            "runtime_seconds": time.perf_counter() - started,
            "finished_at": _utc_now(),
            "artifacts_complete": complete,
            "error": error,
        }
    )
    prepared_manifest = task.artifact_dir / "prepared_data" / "baseline_data_manifest.json"
    if prepared_manifest.is_file():
        try:
            baseline_data = json.loads(prepared_manifest.read_text(encoding="utf-8"))
            record["prepared_data_hashes"] = baseline_data.get("array_hashes")
        except (OSError, json.JSONDecodeError):
            record["prepared_data_hashes"] = None
    if return_code == 0 and not complete:
        record["error"] = "command exited successfully but required artifacts are incomplete"
    _atomic_json(record_path, record)
    return record


def execute_tasks(
    tasks: Iterable[RunSpec],
    *,
    repo_root: Path,
    max_parallel: int = 1,
    resume: bool = False,
    failed_only: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    selected = list(tasks)
    if failed_only:
        selected = [
            task
            for task in selected
            if (task.artifact_dir / "run_record.json").is_file()
            and json.loads(
                (task.artifact_dir / "run_record.json").read_text(encoding="utf-8")
            ).get("status")
            == "failed"
        ]
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    selected_by_id = {task.run_id: task for task in selected}
    pending = dict(selected_by_id)
    states: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    while pending:
        blocked = [
            task
            for task in pending.values()
            if any(states.get(dep) == "failed" for dep in task.dependencies)
        ]
        for task in blocked:
            record = {
                **task.as_dict(),
                "schema_version": 1,
                "status": "failed",
                "artifacts_complete": False,
                "error": "dependency failed",
                "started_at": None,
                "finished_at": _utc_now(),
                "return_code": None,
                "config_hash": hashlib.sha256(
                    canonical_json(task.config).encode("utf-8")
                ).hexdigest(),
                "source_hash": source_fingerprint(str(repo_root.resolve())),
                "environment": environment_snapshot(repo_root),
            }
            record["data_hash"], record["data_paths"] = data_fingerprint(
                task, repo_root
            )
            _atomic_json(task.artifact_dir / "run_record.json", record)
            results.append(record)
            states[task.run_id] = "failed"
            pending.pop(task.run_id)
        if not pending:
            break
        ready = [
            task
            for task in pending.values()
            if all(
                dependency not in selected_by_id
                or states.get(dependency) in {"completed", "skipped_complete", "planned"}
                for dependency in task.dependencies
            )
        ]
        if not ready:
            unresolved = {
                task.run_id: [
                    dep
                    for dep in task.dependencies
                    if dep in selected_by_id and dep not in states
                ]
                for task in pending.values()
            }
            raise ValueError(f"Experiment dependency cycle or unresolved dependency: {unresolved}")
        batch = ready[:max_parallel]
        if max_parallel == 1:
            batch_results = [
                execute_task(batch[0], repo_root=repo_root, resume=resume, dry_run=dry_run)
            ]
        else:
            batch_results = []
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(
                        execute_task,
                        task,
                        repo_root=repo_root,
                        resume=resume,
                        dry_run=dry_run,
                    ): task
                    for task in batch
                }
                for future in as_completed(futures):
                    batch_results.append(future.result())
        for result in batch_results:
            run_id = str(result["run_id"])
            states[run_id] = str(result["status"])
            pending.pop(run_id)
            results.append(result)
    return results
