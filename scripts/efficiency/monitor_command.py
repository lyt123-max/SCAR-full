from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[2]
RESOURCE_MONITOR_PATH = ROOT / "coremad" / "resource_monitor.py"


def _load_resource_monitor_module():
    spec = importlib.util.spec_from_file_location(
        "scar_standalone_resource_monitor",
        RESOURCE_MONITOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load resource monitor from {RESOURCE_MONITOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RESOURCE_MONITOR_MODULE = _load_resource_monitor_module()
ResourceMonitor = _RESOURCE_MONITOR_MODULE.ResourceMonitor
resolve_cuda_device_selector = _RESOURCE_MONITOR_MODULE.resolve_cuda_device_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor an external baseline command without modifying its source tree."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=["train", "build", "inference"], required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--windows", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--stdout-log", type=Path, default=None)
    parser.add_argument("--stderr-log", type=Path, default=None)
    parser.add_argument("--invocation-id", default=None)
    parser.add_argument(
        "--strict-dependencies",
        type=int,
        default=1,
        choices=[0, 1],
        help="Refuse to launch when psutil or required CUDA NVML support is unavailable.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.timeout < 0.0:
        parser.error("--timeout must be non-negative")
    for name in ("points", "windows", "batches"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name} must be non-negative")
    return args


def _dependency_error(device: str) -> str | None:
    try:
        import psutil  # noqa: F401
    except Exception as exc:
        return f"psutil is required for formal monitoring: {type(exc).__name__}: {exc}"
    if not str(device).lower().startswith("cuda"):
        return None
    try:
        import pynvml

        initialized = False
        try:
            pynvml.nvmlInit()
            initialized = True
            selector_type, selector, _ = resolve_cuda_device_selector(device)
            if selector_type == "uuid":
                pynvml.nvmlDeviceGetHandleByUUID(selector)
            else:
                pynvml.nvmlDeviceGetHandleByIndex(int(selector))
        finally:
            if initialized:
                pynvml.nvmlShutdown()
    except Exception as exc:
        return f"NVML is required for formal CUDA monitoring: {type(exc).__name__}: {exc}"
    return None


def _tee_stream(
    source: TextIO,
    destination: TextIO,
    console: TextIO,
    tail: deque[str] | None = None,
) -> None:
    try:
        for chunk in iter(source.readline, ""):
            if tail is not None:
                tail.append(chunk)
            destination.write(chunk)
            destination.flush()
            console.write(chunk)
            console.flush()
    except (OSError, ValueError):
        pass
    finally:
        try:
            source.close()
        except OSError:
            pass


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        import psutil
    except ImportError:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                )
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                try:
                    process.kill()
                except OSError:
                    pass
        return
    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(children, timeout=3.0)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        try:
            root.terminate()
            root.wait(timeout=3.0)
        except psutil.Error:
            try:
                root.kill()
            except psutil.Error:
                pass
    except (psutil.Error, OSError):
        try:
            process.kill()
        except OSError:
            pass


def _wait_after_termination(
    process: subprocess.Popen[str],
    timeout: float = 5.0,
) -> bool:
    try:
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False


def _record_prelaunch_failure(
    *,
    args: argparse.Namespace,
    stdout_path: Path,
    stderr_path: Path,
    invocation_id: str,
    started_perf: float,
    started_at: str,
    message: str,
    return_code: int,
) -> int:
    boundary = f"\n=== invocation {invocation_id} stage={args.stage} ===\n"
    with stdout_path.open("a", encoding="utf-8") as stdout_log:
        stdout_log.write(boundary)
    with stderr_path.open("a", encoding="utf-8") as stderr_log:
        stderr_log.write(boundary + message + "\n")
    print(f"[monitor_command] {message}", file=sys.stderr)
    with ResourceMonitor(
        output_path=args.output,
        method=args.method,
        dataset=args.dataset,
        seed=args.seed,
        stage=args.stage,
        device=args.device,
        invocation_id=invocation_id,
        sample_interval=args.sample_interval,
        command=args.command,
        started_perf=started_perf,
        started_at=started_at,
        sampling_enabled=False,
    ) as monitor:
        monitor.set_status("failed", message)
        monitor.set_process_result(return_code=None)
    return return_code


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = args.stdout_log or args.output.parent / "stdout.log"
    stderr_path = args.stderr_log or args.output.parent / "stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    invocation_id = args.invocation_id or uuid.uuid4().hex
    started_perf = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if args.strict_dependencies:
        dependency_error = _dependency_error(args.device)
        if dependency_error is not None:
            return _record_prelaunch_failure(
                args=args,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                invocation_id=invocation_id,
                started_perf=started_perf,
                started_at=started_at,
                message=dependency_error,
                return_code=2,
            )
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    try:
        process = subprocess.Popen(
            args.command,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
        )
    except OSError as exc:
        message = f"{type(exc).__name__}: {exc}"
        return _record_prelaunch_failure(
            args=args,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            invocation_id=invocation_id,
            started_perf=started_perf,
            started_at=started_at,
            message=message,
            return_code=127,
        )
    assert process.stdout is not None
    assert process.stderr is not None

    with stdout_path.open("a", encoding="utf-8") as stdout_log, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_log:
        boundary = f"\n=== invocation {invocation_id} stage={args.stage} ===\n"
        stdout_log.write(boundary)
        stdout_log.flush()
        stderr_log.write(boundary)
        stderr_log.flush()
        stdout_tail: deque[str] = deque(maxlen=100)
        stderr_tail: deque[str] = deque(maxlen=100)
        stdout_thread = threading.Thread(
            target=_tee_stream,
            args=(process.stdout, stdout_log, sys.stdout, stdout_tail),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_tee_stream,
            args=(process.stderr, stderr_log, sys.stderr, stderr_tail),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        interrupted = False
        terminal_state: str | None = None
        terminal_reason: str | None = None
        with ResourceMonitor(
            output_path=args.output,
            method=args.method,
            dataset=args.dataset,
            seed=args.seed,
            stage=args.stage,
            device=args.device,
            invocation_id=invocation_id,
            sample_interval=args.sample_interval,
            target_pid=process.pid,
            include_children=True,
            command=args.command,
            started_perf=started_perf,
            started_at=started_at,
        ) as monitor:
            workload = {
                name: value
                for name, value in {
                    "points": args.points,
                    "windows": args.windows,
                    "batches": args.batches,
                }.items()
                if value is not None
            }
            monitor.set_workload(**workload)
            try:
                process.wait(timeout=args.timeout if args.timeout > 0.0 else None)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_tree(process)
                terminated = _wait_after_termination(process)
                terminal_state = "timeout"
                terminal_reason = f"command exceeded {args.timeout:g} seconds"
                if not terminated:
                    terminal_reason += "; process did not exit after forced termination"
            except KeyboardInterrupt:
                interrupted = True
                _terminate_process_tree(process)
                terminated = _wait_after_termination(process)
                terminal_state = "interrupted"
                terminal_reason = "monitor interrupted by user"
                if not terminated:
                    terminal_reason += "; process did not exit after forced termination"
            if process.poll() is not None:
                stdout_thread.join()
                stderr_thread.join()
            else:
                stdout_thread.join(timeout=1.0)
                stderr_thread.join(timeout=1.0)
                if stdout_thread.is_alive() or stderr_thread.is_alive():
                    terminal_reason = (
                        f"{terminal_reason}; logs may be truncated"
                        if terminal_reason
                        else "logs may be truncated because the process is still running"
                    )
            if terminal_state is not None:
                monitor.set_status(terminal_state, terminal_reason)
            elif process.returncode != 0:
                stderr_text = "".join(stderr_tail).lower()
                if "out of memory" in stderr_text:
                    monitor.set_status(
                        "oom",
                        f"command reported out of memory and exited with return code {process.returncode}",
                    )
                else:
                    monitor.set_status(
                        "failed",
                        f"command exited with return code {process.returncode}",
                    )
            monitor.set_process_result(
                return_code=process.returncode,
                timed_out=timed_out,
            )

    if timed_out:
        return 124
    if interrupted:
        return 130
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
