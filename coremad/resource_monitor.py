from __future__ import annotations

import copy
import importlib
import json
import os
import platform
import sys
import threading
import time
import uuid
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_STORE_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _exclusive_file_lock(lock_path: Path, timeout: float = 60.0):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        deadline = time.monotonic() + timeout
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out locking {lock_path}")
                    time.sleep(0.05)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out locking {lock_path}")
                    time.sleep(0.05)
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _optional_import(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, f"{name} unavailable: {type(exc).__name__}: {exc}"


def _empty_throughput() -> dict[str, float | None]:
    return {
        "points_per_second": None,
        "covered_points_per_second": None,
        "window_points_per_second": None,
        "input_points_per_second": None,
        "windows_per_second": None,
        "batches_per_second": None,
        "milliseconds_per_window": None,
    }


def resolve_cuda_device_selector(device: str) -> tuple[str, int | str, str]:
    logical_index = ResourceMonitor._resolve_gpu_index(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "index", logical_index, "logical_device_index"
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if logical_index >= len(entries):
        raise ValueError(
            f"{device!r} is outside CUDA_VISIBLE_DEVICES={visible!r}"
        )
    token = entries[logical_index]
    if token.lstrip("+-").isdigit():
        return "index", int(token), "CUDA_VISIBLE_DEVICES"
    if token.startswith(("GPU-", "MIG-")):
        return "uuid", token, "CUDA_VISIBLE_DEVICES"
    raise ValueError(
        f"unsupported CUDA_VISIBLE_DEVICES entry {token!r}; use a physical index or UUID"
    )


def _throughput(workload: Mapping[str, Any], seconds: float) -> dict[str, float | None]:
    result = _empty_throughput()
    if seconds <= 0.0:
        return result
    for unit in (
        "points",
        "covered_points",
        "window_points",
        "input_points",
        "windows",
        "batches",
    ):
        value = workload.get(unit)
        if isinstance(value, (int, float)) and value >= 0:
            result[f"{unit}_per_second"] = float(value) / seconds
    windows = workload.get("windows")
    if isinstance(windows, (int, float)) and windows > 0:
        result["milliseconds_per_window"] = seconds * 1000.0 / float(windows)
    return result


class ResourceMetricsStore:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def upsert(self, attempt: Mapping[str, Any]) -> dict[str, Any]:
        with _STORE_LOCK:
            lock_path = self.output_path.with_name(f".{self.output_path.name}.lock")
            with _exclusive_file_lock(lock_path):
                payload = self._read()
                attempts = list(payload.get("attempts", []))
                attempt_id = attempt["attempt_id"]
                for index, current in enumerate(attempts):
                    if current.get("attempt_id") == attempt_id:
                        attempts[index] = dict(attempt)
                        break
                else:
                    attempts.append(dict(attempt))
                payload["schema_version"] = 1
                payload["updated_at"] = _utc_now()
                payload["attempts"] = attempts
                payload["summary"] = self._summarize(attempts)
                self._atomic_write(payload)
                return payload

    def _read(self) -> dict[str, Any]:
        if not self.output_path.exists():
            return {
                "schema_version": 1,
                "created_at": _utc_now(),
                "attempts": [],
                "summary": {},
            }
        try:
            loaded = json.loads(self.output_path.read_text(encoding="utf-8"))
        except OSError:
            raise
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid existing metrics JSON at {self.output_path}: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"invalid existing metrics payload at {self.output_path}: expected object"
            )
        return loaded

    def _atomic_write(self, payload: Mapping[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(
            f".{self.output_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.output_path)

    @staticmethod
    def _summarize(attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not attempts:
            return {}
        invocation_id = str(attempts[-1].get("invocation_id", ""))
        selected = [
            attempt
            for attempt in attempts
            if attempt.get("invocation_id") == invocation_id
            and attempt.get("status", {}).get("state") == "completed"
            and not attempt.get("status", {}).get("excluded_from_summary", False)
        ]
        summary: dict[str, Any] = {
            "invocation_id": invocation_id,
            "completed_stages": [attempt.get("stage") for attempt in selected],
            "end_to_end_seconds": sum(
                float(attempt.get("timing", {}).get("wall_seconds") or 0.0)
                for attempt in selected
            ),
            "train_seconds": None,
            "build_seconds": None,
            "inference_seconds": None,
        }
        for attempt in selected:
            stage = attempt.get("stage")
            spans = attempt.get("spans", {})
            wall_seconds = float(attempt.get("timing", {}).get("wall_seconds") or 0.0)
            if stage in {"stage_a", "train"}:
                summary["train_seconds"] = float(
                    spans.get("train_loop", {}).get("wall_seconds") or wall_seconds
                )
            elif stage in {"stage_b", "build"}:
                build_spans = [
                    spans.get(name, {}).get("wall_seconds")
                    for name in ("memory_build", "fusion_fit")
                ]
                available = [float(value) for value in build_spans if value is not None]
                summary["build_seconds"] = sum(available) if available else wall_seconds
            elif stage in {"test", "inference"}:
                summary["inference_seconds"] = float(
                    spans.get("inference", {}).get("wall_seconds") or wall_seconds
                )
        return summary


class ResourceSpan(AbstractContextManager["ResourceSpan"]):
    def __init__(self, monitor: "ResourceMonitor", name: str) -> None:
        self.monitor = monitor
        self.name = name
        self.workload: dict[str, int | float] = {}
        self._started: float | None = None

    def __enter__(self) -> "ResourceSpan":
        self.monitor._cuda_synchronize()
        self._started = time.perf_counter()
        return self

    def set_workload(self, **values: int | float) -> None:
        self.workload.update(_validated_workload(values))

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.monitor._cuda_synchronize()
        ended = time.perf_counter()
        seconds = max(0.0, ended - (self._started or ended))
        previous = self.monitor._attempt["spans"].get(self.name, {})
        combined_workload = dict(previous.get("workload", {}))
        for key, value in self.workload.items():
            combined_workload[key] = combined_workload.get(key, 0) + value
        combined_seconds = float(previous.get("wall_seconds") or 0.0) + seconds
        self.monitor._attempt["spans"][self.name] = {
            "wall_seconds": combined_seconds,
            "workload": combined_workload,
            "throughput": _throughput(combined_workload, combined_seconds),
            "sample_count": int(previous.get("sample_count") or 0) + 1,
            "status": (
                "failed"
                if exc_type is not None or previous.get("status") == "failed"
                else "completed"
            ),
        }
        return False


def _validated_workload(values: Mapping[str, int | float]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, value in values.items():
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"workload {key!r} must be a non-negative number")
        result[str(key)] = value
    return result


class ResourceMonitor(AbstractContextManager["ResourceMonitor"]):
    def __init__(
        self,
        *,
        output_path: str | Path,
        method: str,
        dataset: str,
        seed: int,
        stage: str,
        device: str,
        invocation_id: str | None = None,
        sample_interval: float = 0.1,
        target_pid: int | None = None,
        include_children: bool = True,
        command: list[str] | None = None,
        started_perf: float | None = None,
        started_at: str | None = None,
        sampling_enabled: bool = True,
    ) -> None:
        if sample_interval <= 0.0:
            raise ValueError("sample_interval must be positive")
        self.output_path = Path(output_path)
        self.sample_interval = float(sample_interval)
        self.target_pid = int(target_pid or os.getpid())
        self.include_children = bool(include_children)
        self.sampling_enabled = bool(sampling_enabled)
        self._provided_started_perf = started_perf
        self._provided_started_at = started_at
        self._store = ResourceMetricsStore(self.output_path)
        self._stop_event = threading.Event()
        self._sample_lock = threading.RLock()
        self._sampler_thread: threading.Thread | None = None
        self._started_perf: float | None = None
        self._cpu_last_times: dict[int, tuple[float, float]] = {}
        self._cpu_accumulated_user = 0.0
        self._cpu_accumulated_system = 0.0
        self._cpu_sampled_once = False
        self._requested_status: tuple[str, str | None] | None = None
        self._psutil, psutil_error = _optional_import("psutil")
        self._torch: Any | None = None
        self._nvml: Any | None = None
        self._nvml_handle: Any | None = None
        self._gpu_index: int | None = None
        self._device = str(device)
        self._attempt: dict[str, Any] = {
            "attempt_id": uuid.uuid4().hex,
            "invocation_id": invocation_id or uuid.uuid4().hex,
            "method": str(method),
            "dataset": str(dataset),
            "seed": int(seed),
            "stage": str(stage),
            "device": self._device,
            "command": list(command) if command else None,
            "pid": self.target_pid,
            "process": {
                "return_code": None,
                "timed_out": False,
            },
            "timing": {
                "started_at": None,
                "finished_at": None,
                "wall_seconds": None,
                "cpu_user_seconds": None,
                "cpu_system_seconds": None,
            },
            "cpu": {
                "available": self._psutil is not None and self.sampling_enabled,
                "reason": (
                    psutil_error
                    if self.sampling_enabled
                    else "sampling disabled before target process started"
                ),
                "peak_rss_bytes": None,
                "sample_count": 0,
                "sample_interval_seconds": self.sample_interval,
                "includes_children": self.include_children,
            },
            "gpu": {
                "available": False,
                "reason": None,
                "device_index": None,
                "physical_device_index": None,
                "device_identifier": None,
                "mapping_source": None,
                "start_device_used_bytes": None,
                "peak_device_used_bytes": None,
                "peak_process_used_bytes": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "sample_count": 0,
            },
            "workload": {},
            "throughput": _empty_throughput(),
            "model": {
                "total_parameters": None,
                "trainable_parameters": None,
            },
            "artifacts": {},
            "artifact_total_bytes": 0,
            "spans": {},
            "environment": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "warnings": [],
            "status": {
                "state": "running",
                "reason": None,
                "exception_type": None,
                "exception_message": None,
                "excluded_from_summary": False,
            },
        }

    @property
    def invocation_id(self) -> str:
        return str(self._attempt["invocation_id"])

    def __enter__(self) -> "ResourceMonitor":
        self._attempt["timing"]["started_at"] = self._provided_started_at or _utc_now()
        self._started_perf = self._provided_started_perf or time.perf_counter()
        if self.sampling_enabled:
            self._initialize_gpu()
            self._sample_once()
        else:
            self._attempt["gpu"]["reason"] = "sampling disabled before target process started"
        self._persist_attempt()
        if self.sampling_enabled:
            self._sampler_thread = threading.Thread(
                target=self._sample_loop,
                name=f"resource-monitor-{self._attempt['attempt_id'][:8]}",
                daemon=True,
            )
            self._sampler_thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=max(5.0, self.sample_interval * 3.0))
            if self._sampler_thread.is_alive():
                reason = (
                    "sampler thread did not stop; NVML shutdown was deferred to avoid "
                    "concurrent sampler access"
                )
                self._warn(reason)
                snapshot = copy.deepcopy(self._attempt)
                finished_perf = time.perf_counter()
                snapshot["timing"]["finished_at"] = _utc_now()
                snapshot["timing"]["wall_seconds"] = max(
                    0.0,
                    finished_perf - (self._started_perf or finished_perf),
                )
                snapshot["status"].update(
                    {
                        "state": "failed",
                        "reason": reason,
                        "excluded_from_summary": True,
                    }
                )
                try:
                    self._store.upsert(snapshot)
                except Exception as exc:
                    self._warn(
                        f"resource metrics persistence failed: {type(exc).__name__}: {exc}"
                    )
                threading.Thread(
                    target=self._finish_deferred_sampler_cleanup,
                    name=f"resource-monitor-cleanup-{self._attempt['attempt_id'][:8]}",
                    daemon=True,
                ).start()
                return False
        if self.sampling_enabled:
            self._sample_once()
            self._cuda_synchronize()
            self._collect_torch_peaks()
            self._shutdown_nvml()
        finished_perf = time.perf_counter()
        self._attempt["timing"]["finished_at"] = _utc_now()
        self._attempt["timing"]["wall_seconds"] = max(
            0.0, finished_perf - (self._started_perf or finished_perf)
        )
        self._finalize_cpu_times()
        self._attempt["throughput"] = _throughput(
            self._attempt["workload"],
            float(self._attempt["timing"]["wall_seconds"]),
        )
        if exc_type is not None:
            message = str(exc)
            if issubclass(exc_type, KeyboardInterrupt):
                state = "interrupted"
            elif "out of memory" in message.lower():
                state = "oom"
            else:
                state = "failed"
            self._attempt["status"].update(
                {
                    "state": state,
                    "reason": message,
                    "exception_type": exc_type.__name__,
                    "exception_message": message,
                    "excluded_from_summary": True,
                }
            )
        elif self._requested_status is not None:
            state, reason = self._requested_status
            self._attempt["status"].update(
                {
                    "state": state,
                    "reason": reason,
                    "excluded_from_summary": state != "completed",
                }
            )
        else:
            self._attempt["status"]["state"] = "completed"
        self._persist_attempt()
        return False

    def span(self, name: str) -> ResourceSpan:
        if not name:
            raise ValueError("span name must be non-empty")
        return ResourceSpan(self, name)

    def set_workload(self, **values: int | float) -> None:
        self._attempt["workload"].update(_validated_workload(values))

    def add_workload(self, **values: int | float) -> None:
        for key, value in _validated_workload(values).items():
            self._attempt["workload"][key] = self._attempt["workload"].get(key, 0) + value

    def set_model_stats(self, *, total_parameters: int, trainable_parameters: int) -> None:
        if total_parameters < 0 or trainable_parameters < 0:
            raise ValueError("parameter counts must be non-negative")
        self._attempt["model"] = {
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
        }

    def set_artifact_bytes(self, **values: int | Path | str | None) -> None:
        for name, value in values.items():
            if value is None:
                size = None
            elif isinstance(value, int):
                if value < 0:
                    raise ValueError(f"artifact size {name!r} must be non-negative")
                size = value
            else:
                path = Path(value)
                try:
                    size = path.stat().st_size if path.is_file() else None
                except OSError as exc:
                    size = None
                    self._warn(f"artifact stat failed for {path}: {type(exc).__name__}: {exc}")
            self._attempt["artifacts"][str(name)] = size
        self._attempt["artifact_total_bytes"] = sum(
            int(size)
            for size in self._attempt["artifacts"].values()
            if isinstance(size, int)
        )

    def set_process_result(self, *, return_code: int | None, timed_out: bool = False) -> None:
        self._attempt["process"] = {
            "return_code": None if return_code is None else int(return_code),
            "timed_out": bool(timed_out),
        }

    def mark_skipped(self, reason: str) -> None:
        self.set_status("skipped", reason)

    def set_status(self, state: str, reason: str | None = None) -> None:
        allowed = {"completed", "skipped", "failed", "oom", "timeout", "interrupted"}
        if state not in allowed:
            raise ValueError(f"unsupported resource monitor status: {state}")
        self._requested_status = (state, reason)

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval):
            self._sample_once()

    def _finish_deferred_sampler_cleanup(self) -> None:
        if self._sampler_thread is None:
            return
        self._sampler_thread.join()
        with self._sample_lock:
            self._shutdown_nvml()

    def _persist_attempt(self) -> None:
        try:
            self._store.upsert(self._attempt)
        except Exception as exc:
            self._warn(f"resource metrics persistence failed: {type(exc).__name__}: {exc}")

    def _warn(self, message: str) -> None:
        warnings = self._attempt.setdefault("warnings", [])
        if message not in warnings:
            warnings.append(message)
            print(f"[ResourceMonitor] warning: {message}", file=sys.stderr)

    def _sample_once(self) -> None:
        with self._sample_lock:
            self._sample_once_locked()

    def _sample_once_locked(self) -> None:
        pids = {self.target_pid}
        if self._psutil is not None:
            try:
                root = self._psutil.Process(self.target_pid)
                processes = [root]
                if self.include_children:
                    processes.extend(root.children(recursive=True))
                pids.update(int(process.pid) for process in processes)
                rss = 0
                for process in processes:
                    try:
                        rss += int(process.memory_info().rss)
                        cpu_times = process.cpu_times()
                        process_pid = int(process.pid)
                        current_user = float(cpu_times.user)
                        current_system = float(cpu_times.system)
                        previous = self._cpu_last_times.get(process_pid)
                        if previous is not None:
                            self._cpu_accumulated_user += max(
                                0.0,
                                current_user - previous[0],
                            )
                            self._cpu_accumulated_system += max(
                                0.0,
                                current_system - previous[1],
                            )
                        elif self._cpu_sampled_once or self.target_pid != os.getpid():
                            self._cpu_accumulated_user += max(0.0, current_user)
                            self._cpu_accumulated_system += max(0.0, current_system)
                        self._cpu_last_times[process_pid] = (
                            current_user,
                            current_system,
                        )
                    except Exception:
                        continue
                cpu = self._attempt["cpu"]
                cpu["peak_rss_bytes"] = max(int(cpu["peak_rss_bytes"] or 0), rss)
                cpu["sample_count"] += 1
                self._cpu_sampled_once = True
            except Exception as exc:
                cpu = self._attempt["cpu"]
                process_finished = (
                    type(exc).__name__ == "NoSuchProcess"
                    and int(cpu.get("sample_count") or 0) > 0
                )
                if not process_finished:
                    cpu["reason"] = (
                        f"process sampling failed: {type(exc).__name__}: {exc}"
                    )
        self._sample_gpu(pids)

    def _initialize_gpu(self) -> None:
        gpu = self._attempt["gpu"]
        if not self._device.lower().startswith("cuda"):
            gpu["reason"] = f"device {self._device!r} is not CUDA"
            return
        torch_error: str | None
        if self.target_pid != os.getpid():
            torch_error = "PyTorch allocator metrics are only available for in-process monitoring"
        else:
            self._torch, torch_error = _optional_import("torch")
            if self._torch is not None:
                try:
                    if not self._torch.cuda.is_available():
                        torch_error = "torch.cuda.is_available() is false"
                        self._torch = None
                    else:
                        self._gpu_index = self._resolve_gpu_index(self._device)
                        self._torch.cuda.synchronize(self._gpu_index)
                        self._torch.cuda.reset_peak_memory_stats(self._gpu_index)
                        gpu["available"] = True
                        gpu["device_index"] = self._gpu_index
                except Exception as exc:
                    torch_error = f"PyTorch CUDA initialization failed: {type(exc).__name__}: {exc}"
                    self._torch = None
        self._nvml, nvml_error = _optional_import("pynvml")
        if self._nvml is not None:
            try:
                self._nvml.nvmlInit()
                self._gpu_index = (
                    self._gpu_index
                    if self._gpu_index is not None
                    else self._resolve_gpu_index(self._device)
                )
                self._nvml_handle = self._resolve_nvml_handle()
                start_used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used)
                gpu["available"] = True
                gpu["device_index"] = self._gpu_index
                gpu["start_device_used_bytes"] = start_used
                gpu["peak_device_used_bytes"] = start_used
            except Exception as exc:
                nvml_error = f"NVML initialization failed: {type(exc).__name__}: {exc}"
                self._shutdown_nvml()
        reasons = [reason for reason in (torch_error, nvml_error) if reason]
        gpu["reason"] = "; ".join(reasons) if reasons else None

    @staticmethod
    def _resolve_gpu_index(device: str) -> int:
        _, separator, suffix = device.partition(":")
        return int(suffix) if separator and suffix else 0

    def _resolve_nvml_handle(self) -> Any:
        gpu = self._attempt["gpu"]
        if self._torch is not None:
            get_properties = getattr(self._torch.cuda, "get_device_properties", None)
            get_by_uuid = getattr(self._nvml, "nvmlDeviceGetHandleByUUID", None)
            if get_properties is not None and get_by_uuid is not None:
                properties = get_properties(self._gpu_index)
                device_uuid = getattr(properties, "uuid", None)
                if device_uuid:
                    identifier = (
                        device_uuid.decode("utf-8")
                        if isinstance(device_uuid, bytes)
                        else str(device_uuid)
                    )
                    handle = get_by_uuid(identifier)
                    gpu["device_identifier"] = identifier
                    gpu["mapping_source"] = "torch_device_uuid"
                    get_index = getattr(self._nvml, "nvmlDeviceGetIndex", None)
                    if get_index is not None:
                        gpu["physical_device_index"] = int(get_index(handle))
                    return handle
        selector_type, selector, source = resolve_cuda_device_selector(self._device)
        gpu["device_identifier"] = selector
        gpu["mapping_source"] = source
        if selector_type == "uuid":
            getter = getattr(self._nvml, "nvmlDeviceGetHandleByUUID", None)
            if getter is None:
                raise RuntimeError("installed NVML bindings cannot resolve GPU UUIDs")
            handle = getter(selector)
        else:
            handle = self._nvml.nvmlDeviceGetHandleByIndex(int(selector))
            gpu["physical_device_index"] = int(selector)
        get_index = getattr(self._nvml, "nvmlDeviceGetIndex", None)
        if get_index is not None:
            gpu["physical_device_index"] = int(get_index(handle))
        return handle

    def _cuda_synchronize(self) -> None:
        if self._torch is None or self._gpu_index is None:
            return
        try:
            self._torch.cuda.synchronize(self._gpu_index)
        except Exception as exc:
            self._attempt["gpu"]["reason"] = (
                f"CUDA synchronization failed: {type(exc).__name__}: {exc}"
            )

    def _sample_gpu(self, pids: set[int]) -> None:
        if self._nvml is None or self._nvml_handle is None:
            return
        gpu = self._attempt["gpu"]
        try:
            device_used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used)
            gpu["peak_device_used_bytes"] = max(
                int(gpu["peak_device_used_bytes"] or 0), device_used
            )
            process_memory: dict[int, int] = {}
            unavailable_seen = False
            unavailable_value = getattr(self._nvml, "NVML_VALUE_NOT_AVAILABLE", None)
            for getter_name in (
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                getter = getattr(self._nvml, getter_name, None)
                if getter is None:
                    continue
                for process in getter(self._nvml_handle):
                    process_pid = int(process.pid)
                    if process_pid in pids:
                        used = getattr(process, "usedGpuMemory", 0)
                        if unavailable_value is not None and used == unavailable_value:
                            unavailable_seen = True
                            continue
                        if isinstance(used, int) and used >= 0:
                            process_memory[process_pid] = max(
                                process_memory.get(process_pid, 0),
                                used,
                            )
            if process_memory:
                process_used = sum(process_memory.values())
                gpu["peak_process_used_bytes"] = max(
                    int(gpu["peak_process_used_bytes"] or 0), process_used
                )
            elif unavailable_seen:
                reason = "NVML_VALUE_NOT_AVAILABLE returned for process GPU memory"
                if reason not in str(gpu.get("reason") or ""):
                    gpu["reason"] = "; ".join(
                        value for value in (gpu.get("reason"), reason) if value
                    )
            else:
                gpu["peak_process_used_bytes"] = max(
                    int(gpu["peak_process_used_bytes"] or 0),
                    0,
                )
            gpu["sample_count"] += 1
        except Exception as exc:
            gpu["reason"] = f"NVML sampling failed: {type(exc).__name__}: {exc}"

    def _collect_torch_peaks(self) -> None:
        if self._torch is None or self._gpu_index is None:
            return
        gpu = self._attempt["gpu"]
        try:
            gpu["peak_allocated_bytes"] = int(
                self._torch.cuda.max_memory_allocated(self._gpu_index)
            )
            gpu["peak_reserved_bytes"] = int(
                self._torch.cuda.max_memory_reserved(self._gpu_index)
            )
        except Exception as exc:
            gpu["reason"] = f"PyTorch peak query failed: {type(exc).__name__}: {exc}"

    def _finalize_cpu_times(self) -> None:
        if int(self._attempt["cpu"].get("sample_count") or 0) <= 0:
            return
        self._attempt["timing"]["cpu_user_seconds"] = self._cpu_accumulated_user
        self._attempt["timing"]["cpu_system_seconds"] = self._cpu_accumulated_system

    def _shutdown_nvml(self) -> None:
        if self._nvml is None:
            return
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
        finally:
            self._nvml = None
            self._nvml_handle = None
