from __future__ import annotations

import importlib.util
import ast
import json
import os
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


MODULE_PATH = Path(__file__).resolve().parents[1] / "coremad" / "resource_monitor.py"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "coremad" / "config.py"
COMMAND_MONITOR_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "efficiency" / "monitor_command.py"
)
TRAINER_PATH = Path(__file__).resolve().parents[1] / "coremad" / "trainer.py"
RUN_PATH = Path(__file__).resolve().parents[1] / "run.py"


def load_resource_monitor_module():
    spec = importlib.util.spec_from_file_location("scar_resource_monitor", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load resource monitor from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config_module():
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = torch_stub
    try:
        spec = importlib.util.spec_from_file_location("scar_config_for_monitor_test", CONFIG_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load config from {CONFIG_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch


def load_command_monitor_module():
    spec = importlib.util.spec_from_file_location(
        "scar_command_monitor_for_test",
        COMMAND_MONITOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load command monitor from {COMMAND_MONITOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResourceMonitorTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("psutil"), "psutil is not installed")
    def test_command_wrapper_samples_real_child_rss(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMMAND_MONITOR_PATH),
                    "--output",
                    str(output),
                    "--method",
                    "PGRF-Net",
                    "--dataset",
                    "MSL",
                    "--seed",
                    "42",
                    "--stage",
                    "train",
                    "--device",
                    "cpu",
                    "--sample-interval",
                    "0.02",
                    "--",
                    sys.executable,
                    "-c",
                    "import time; payload=bytearray(8*1024*1024); time.sleep(0.8)",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )
            self.assertEqual(completed.returncode, 0)
            cpu = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]["cpu"]
            self.assertTrue(cpu["available"])
            self.assertGreater(cpu["sample_count"], 1)
            self.assertGreater(cpu["peak_rss_bytes"], 8 * 1024 * 1024)
            self.assertIsNone(cpu["reason"])

    def test_trainer_and_cli_expose_monitoring_contract(self) -> None:
        trainer_tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
        trainer_methods = {
            node.name
            for node in ast.walk(trainer_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(
            {
                "_run_monitored_stage",
                "_run_stage_a_impl",
                "_run_stage_b_impl",
                "_run_test_impl",
            }.issubset(trainer_methods)
        )
        run_full_node = next(
            node
            for node in ast.walk(trainer_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_full"
        )
        run_full_calls = {
            node.func.attr
            for node in ast.walk(run_full_node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("run_stage_b", run_full_calls)
        self.assertNotIn("_is_stage_b_complete", run_full_calls)
        trainer_source = TRAINER_PATH.read_text(encoding="utf-8")
        self.assertIn("ResourceMonitor", trainer_source)
        self.assertIn('_resource_span("train_loop")', trainer_source)
        self.assertIn('_resource_span("memory_build")', trainer_source)
        self.assertIn('_resource_span("fusion_fit")', trainer_source)
        self.assertIn('_resource_span("inference")', trainer_source)
        self.assertIn('_resource_span("scoring")', trainer_source)
        self.assertIn('_resource_span("evaluation")', trainer_source)
        self.assertIn('_resource_span("result_export")', trainer_source)
        self.assertIn('"resource_monitor_enabled"', trainer_source)
        self.assertIn('"resource_sample_interval"', trainer_source)

        run_tree = ast.parse(RUN_PATH.read_text(encoding="utf-8"))
        option_names = {
            node.args[0].value
            for node in ast.walk(run_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        self.assertIn("--resource_monitor", option_names)
        self.assertIn("--resource_sample_interval", option_names)
        aggregate_node = next(
            node
            for node in ast.walk(trainer_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_aggregate_point_diagnostics"
        )
        aggregate_span_names = {
            node.args[0].value
            for node in ast.walk(aggregate_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_resource_span"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        self.assertIn("inference", aggregate_span_names)

    def test_command_wrapper_records_success_and_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMMAND_MONITOR_PATH),
                    "--output",
                    str(output),
                    "--method",
                    "CATCH",
                    "--dataset",
                    "MSL",
                    "--seed",
                    "42",
                    "--stage",
                    "inference",
                    "--device",
                    "cpu",
                    "--points",
                    "100",
                    "--windows",
                    "10",
                    "--",
                    sys.executable,
                    "-c",
                    "import sys; print('wrapper-out'); print('wrapper-err', file=sys.stderr)",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )
            self.assertEqual(completed.returncode, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            attempt = payload["attempts"][0]
            self.assertEqual(attempt["status"]["state"], "completed")
            self.assertEqual(attempt["process"]["return_code"], 0)
            self.assertEqual(attempt["workload"]["points"], 100)
            self.assertIn("wrapper-out", (Path(tmp) / "stdout.log").read_text(encoding="utf-8"))
            self.assertIn("wrapper-err", (Path(tmp) / "stderr.log").read_text(encoding="utf-8"))

    def test_command_wrapper_appends_logs_across_attempts(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            for marker in ("first-attempt", "second-attempt"):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(COMMAND_MONITOR_PATH),
                        "--output",
                        str(output),
                        "--method",
                        "CATCH",
                        "--dataset",
                        "MSL",
                        "--seed",
                        "42",
                        "--stage",
                        "inference",
                        "--device",
                        "cpu",
                        "--",
                        sys.executable,
                        "-c",
                        f"print('{marker}')",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    check=False,
                )
                self.assertEqual(completed.returncode, 0)
            stdout_text = (Path(tmp) / "stdout.log").read_text(encoding="utf-8")
            self.assertIn("first-attempt", stdout_text)
            self.assertIn("second-attempt", stdout_text)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["attempts"]), 2)

    def test_command_wrapper_persists_process_start_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMMAND_MONITOR_PATH),
                    "--output",
                    str(output),
                    "--method",
                    "PUAD",
                    "--dataset",
                    "MSL",
                    "--seed",
                    "42",
                    "--stage",
                    "train",
                    "--device",
                    "cpu",
                    "--",
                    str(Path(tmp) / "missing-command.exe"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 127)
            attempt = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]
            self.assertEqual(attempt["status"]["state"], "failed")
            self.assertIsNone(attempt["process"]["return_code"])
            self.assertIn("FileNotFoundError", attempt["status"]["reason"])

    def test_command_wrapper_preserves_nonzero_exit_code(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMMAND_MONITOR_PATH),
                    "--output",
                    str(output),
                    "--method",
                    "PUAD",
                    "--dataset",
                    "MSL",
                    "--seed",
                    "42",
                    "--stage",
                    "train",
                    "--device",
                    "cpu",
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )
            self.assertEqual(completed.returncode, 7)
            attempt = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]
            self.assertEqual(attempt["status"]["state"], "failed")
            self.assertEqual(attempt["process"]["return_code"], 7)

    def test_command_wrapper_classifies_cuda_oom(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMMAND_MONITOR_PATH),
                    "--output",
                    str(output),
                    "--method",
                    "PaAno",
                    "--dataset",
                    "MSL",
                    "--seed",
                    "42",
                    "--stage",
                    "train",
                    "--device",
                    "cuda:0",
                    "--strict-dependencies",
                    "0",
                    "--",
                    sys.executable,
                    "-c",
                    "import sys; print('CUDA out of memory', file=sys.stderr); raise SystemExit(1)",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            attempt = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]
            self.assertEqual(attempt["status"]["state"], "oom")

    def test_process_tree_termination_tolerates_already_exited_process(self) -> None:
        module = load_command_monitor_module()

        class FakePsutilError(Exception):
            pass

        fake_psutil = types.ModuleType("psutil")
        fake_psutil.Error = FakePsutilError
        fake_psutil.Process = lambda pid: (_ for _ in ()).throw(
            FakePsutilError("already exited")
        )
        previous_psutil = sys.modules.get("psutil")
        sys.modules["psutil"] = fake_psutil
        process = types.SimpleNamespace(pid=123, kill_calls=0)

        def kill():
            process.kill_calls += 1

        process.kill = kill
        try:
            module._terminate_process_tree(process)
        finally:
            if previous_psutil is None:
                sys.modules.pop("psutil", None)
            else:
                sys.modules["psutil"] = previous_psutil
        self.assertEqual(process.kill_calls, 1)

    def test_command_wrapper_marks_timeout_and_returns_124(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMMAND_MONITOR_PATH),
                    "--output",
                    str(output),
                    "--method",
                    "GDFlex",
                    "--dataset",
                    "MSL",
                    "--seed",
                    "42",
                    "--stage",
                    "inference",
                    "--device",
                    "cpu",
                    "--timeout",
                    "0.1",
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(5)",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 124)
            attempt = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]
            self.assertEqual(attempt["status"]["state"], "timeout")
            self.assertTrue(attempt["process"]["timed_out"])

    def test_command_wrapper_strict_mode_rejects_missing_psutil_before_launch(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(COMMAND_MONITOR_PATH),
                    "--output",
                    str(output),
                    "--method",
                    "PUAD",
                    "--dataset",
                    "MSL",
                    "--seed",
                    "42",
                    "--stage",
                    "train",
                    "--device",
                    "cpu",
                    "--",
                    sys.executable,
                    "-c",
                    "print('must-not-run')",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("psutil", completed.stderr)
            self.assertNotIn("must-not-run", completed.stdout)
            attempt = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]
            self.assertEqual(attempt["status"]["state"], "failed")
            self.assertIn("psutil", attempt["status"]["reason"])

    def test_strict_dependency_check_respects_cuda_visible_devices(self) -> None:
        module = load_command_monitor_module()
        fake_psutil = types.ModuleType("psutil")
        nvml_indices: list[int] = []
        fake_nvml = types.ModuleType("pynvml")
        fake_nvml.nvmlInit = lambda: None
        fake_nvml.nvmlShutdown = lambda: None
        fake_nvml.nvmlDeviceGetHandleByIndex = lambda index: nvml_indices.append(index)
        previous_psutil = sys.modules.get("psutil")
        previous_nvml = sys.modules.get("pynvml")
        previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        sys.modules["psutil"] = fake_psutil
        sys.modules["pynvml"] = fake_nvml
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        try:
            self.assertIsNone(module._dependency_error("cuda:0"))
        finally:
            if previous_psutil is None:
                sys.modules.pop("psutil", None)
            else:
                sys.modules["psutil"] = previous_psutil
            if previous_nvml is None:
                sys.modules.pop("pynvml", None)
            else:
                sys.modules["pynvml"] = previous_nvml
            if previous_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
        self.assertEqual(nvml_indices, [3])

    def test_config_enables_monitoring_by_default_and_validates_interval(self) -> None:
        module = load_config_module()
        config = module.CoReMADConfig(artifact_root="artifacts", experiment_name="monitor-test")
        self.assertTrue(config.resource_monitor_enabled)
        self.assertEqual(config.resource_sample_interval, 0.1)
        self.assertEqual(
            config.resource_metrics_path,
            Path("artifacts") / "monitor-test" / "resource_metrics.json",
        )
        with self.assertRaisesRegex(ValueError, "resource_sample_interval"):
            module.CoReMADConfig(resource_sample_interval=0.0)

    def test_records_nested_span_workload_and_merges_attempts(self) -> None:
        module = load_resource_monitor_module()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            with module.ResourceMonitor(
                output_path=output,
                method="SCAR",
                dataset="MSL",
                seed=42,
                stage="stage_a",
                device="cpu",
                invocation_id="invocation-test",
                sample_interval=0.01,
            ) as monitor:
                monitor.set_workload(points=1000, windows=20, batches=4, epochs=2)
                monitor.set_model_stats(total_parameters=120, trainable_parameters=100)
                with monitor.span("train_loop") as span:
                    span.set_workload(points=1000, windows=20, batches=4)
                    time.sleep(0.02)

            with module.ResourceMonitor(
                output_path=output,
                method="SCAR",
                dataset="MSL",
                seed=42,
                stage="stage_b",
                device="cpu",
                invocation_id="invocation-test",
                sample_interval=0.01,
            ):
                time.sleep(0.01)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["attempts"]), 2)
            first = payload["attempts"][0]
            self.assertEqual(first["status"]["state"], "completed")
            self.assertEqual(first["workload"]["points"], 1000)
            self.assertEqual(first["model"]["total_parameters"], 120)
            self.assertGreater(first["spans"]["train_loop"]["wall_seconds"], 0.0)
            self.assertGreater(first["spans"]["train_loop"]["throughput"]["points_per_second"], 0.0)
            self.assertEqual(payload["summary"]["invocation_id"], "invocation-test")
            self.assertGreater(payload["summary"]["end_to_end_seconds"], 0.0)

    def test_repeated_spans_accumulate_time_and_workload(self) -> None:
        module = load_resource_monitor_module()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            with module.ResourceMonitor(
                output_path=output,
                method="SCAR",
                dataset="TEP",
                seed=42,
                stage="test",
                device="cpu",
                sample_interval=0.01,
            ) as monitor:
                for windows in (3, 7):
                    with monitor.span("inference") as span:
                        span.set_workload(
                            points=windows * 128,
                            windows=windows,
                            batches=1,
                        )
                        time.sleep(0.01)

            attempt = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]
            inference = attempt["spans"]["inference"]
            self.assertEqual(inference["workload"]["windows"], 10)
            self.assertEqual(inference["workload"]["points"], 1280)
            self.assertEqual(inference["workload"]["batches"], 2)
            self.assertEqual(inference["sample_count"], 2)
            self.assertGreater(inference["wall_seconds"], 0.015)

    def test_external_train_and_build_stages_fill_summary_fields(self) -> None:
        module = load_resource_monitor_module()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            for stage in ("train", "build"):
                with module.ResourceMonitor(
                    output_path=output,
                    method="PaAno",
                    dataset="MSL",
                    seed=42,
                    stage=stage,
                    device="cpu",
                    invocation_id="external-run",
                    sample_interval=0.01,
                ):
                    time.sleep(0.01)
            summary = json.loads(output.read_text(encoding="utf-8"))["summary"]
            self.assertGreater(summary["train_seconds"], 0.0)
            self.assertGreater(summary["build_seconds"], 0.0)

    def test_concurrent_processes_preserve_all_attempts(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            code = """
import importlib.util
import sys
import time

spec = importlib.util.spec_from_file_location("rm_child", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_read = module.ResourceMetricsStore._read

def slow_read(self):
    payload = original_read(self)
    time.sleep(0.1)
    return payload

module.ResourceMetricsStore._read = slow_read
with module.ResourceMonitor(
    output_path=sys.argv[2],
    method="SCAR",
    dataset="MSL",
    seed=int(sys.argv[3]),
    stage="test",
    device="cpu",
    invocation_id="concurrent",
    sample_interval=0.01,
):
    pass
"""
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        code,
                        str(MODULE_PATH),
                        str(output),
                        str(seed),
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                )
                for seed in range(12)
            ]
            for process in processes:
                self.assertEqual(process.wait(timeout=10), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["attempts"]), 12)
            self.assertEqual(
                {attempt["seed"] for attempt in payload["attempts"]},
                set(range(12)),
            )

    def test_cuda_metrics_use_torch_and_do_not_double_count_nvml_processes(self) -> None:
        module = load_resource_monitor_module()
        cuda_calls: list[tuple[str, int]] = []
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda index: cuda_calls.append(("sync", index)),
            reset_peak_memory_stats=lambda index: cuda_calls.append(("reset", index)),
            max_memory_allocated=lambda index: 123,
            max_memory_reserved=lambda index: 456,
        )
        nvml_stub = types.ModuleType("pynvml")
        nvml_stub.nvmlInit = lambda: None
        nvml_stub.nvmlShutdown = lambda: None
        nvml_stub.nvmlDeviceGetHandleByIndex = lambda index: f"gpu-{index}"
        nvml_stub.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(used=1000)
        process = types.SimpleNamespace(pid=os.getpid(), usedGpuMemory=321)
        nvml_stub.nvmlDeviceGetComputeRunningProcesses = lambda handle: [process]
        nvml_stub.nvmlDeviceGetGraphicsRunningProcesses = lambda handle: [process]
        previous_torch = sys.modules.get("torch")
        previous_nvml = sys.modules.get("pynvml")
        sys.modules["torch"] = torch_stub
        sys.modules["pynvml"] = nvml_stub
        try:
            with TemporaryDirectory() as tmp:
                output = Path(tmp) / "resource_metrics.json"
                with module.ResourceMonitor(
                    output_path=output,
                    method="SCAR",
                    dataset="MSL",
                    seed=42,
                    stage="test",
                    device="cuda:0",
                    sample_interval=0.01,
                ):
                    time.sleep(0.01)
                gpu = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]["gpu"]
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch
            if previous_nvml is None:
                sys.modules.pop("pynvml", None)
            else:
                sys.modules["pynvml"] = previous_nvml

        self.assertIn(("reset", 0), cuda_calls)
        self.assertEqual(gpu["peak_allocated_bytes"], 123)
        self.assertEqual(gpu["peak_reserved_bytes"], 456)
        self.assertEqual(gpu["peak_device_used_bytes"], 1000)
        self.assertEqual(gpu["peak_process_used_bytes"], 321)

    def test_cuda_visible_devices_maps_logical_index_to_nvml_physical_index(self) -> None:
        module = load_resource_monitor_module()
        nvml_indices: list[int] = []
        nvml_stub = types.ModuleType("pynvml")
        nvml_stub.nvmlInit = lambda: None
        nvml_stub.nvmlShutdown = lambda: None

        def handle_by_index(index):
            nvml_indices.append(index)
            return f"gpu-{index}"

        nvml_stub.nvmlDeviceGetHandleByIndex = handle_by_index
        nvml_stub.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(used=100)
        nvml_stub.nvmlDeviceGetComputeRunningProcesses = lambda handle: []
        nvml_stub.nvmlDeviceGetGraphicsRunningProcesses = lambda handle: []
        previous_nvml = sys.modules.get("pynvml")
        previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        sys.modules["pynvml"] = nvml_stub
        os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"
        try:
            with TemporaryDirectory() as tmp:
                output = Path(tmp) / "resource_metrics.json"
                with module.ResourceMonitor(
                    output_path=output,
                    method="CATCH",
                    dataset="MSL",
                    seed=42,
                    stage="inference",
                    device="cuda:0",
                    target_pid=os.getpid() + 100000,
                    sample_interval=0.01,
                ):
                    pass
                gpu = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]["gpu"]
        finally:
            if previous_nvml is None:
                sys.modules.pop("pynvml", None)
            else:
                sys.modules["pynvml"] = previous_nvml
            if previous_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible

        self.assertTrue(nvml_indices)
        self.assertTrue(all(index == 3 for index in nvml_indices))
        self.assertEqual(gpu["physical_device_index"], 3)
        self.assertEqual(gpu["mapping_source"], "CUDA_VISIBLE_DEVICES")

    def test_nvml_unavailable_memory_sentinel_is_not_reported_as_peak(self) -> None:
        module = load_resource_monitor_module()
        unavailable = 2**64 - 1
        nvml_stub = types.ModuleType("pynvml")
        nvml_stub.NVML_VALUE_NOT_AVAILABLE = unavailable
        nvml_stub.nvmlInit = lambda: None
        nvml_stub.nvmlShutdown = lambda: None
        nvml_stub.nvmlDeviceGetHandleByIndex = lambda index: f"gpu-{index}"
        nvml_stub.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(used=100)
        target_pid = os.getpid() + 100000
        process = types.SimpleNamespace(pid=target_pid, usedGpuMemory=unavailable)
        nvml_stub.nvmlDeviceGetComputeRunningProcesses = lambda handle: [process]
        nvml_stub.nvmlDeviceGetGraphicsRunningProcesses = lambda handle: []
        previous_nvml = sys.modules.get("pynvml")
        sys.modules["pynvml"] = nvml_stub
        try:
            with TemporaryDirectory() as tmp:
                output = Path(tmp) / "resource_metrics.json"
                with module.ResourceMonitor(
                    output_path=output,
                    method="CATCH",
                    dataset="MSL",
                    seed=42,
                    stage="inference",
                    device="cuda:0",
                    target_pid=target_pid,
                    sample_interval=0.01,
                ):
                    pass
                gpu = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]["gpu"]
        finally:
            if previous_nvml is None:
                sys.modules.pop("pynvml", None)
            else:
                sys.modules["pynvml"] = previous_nvml

        self.assertIsNone(gpu["peak_process_used_bytes"])
        self.assertIn("NVML_VALUE_NOT_AVAILABLE", gpu["reason"])

    def test_external_process_never_reports_monitor_torch_allocator(self) -> None:
        module = load_resource_monitor_module()
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda index: None,
            reset_peak_memory_stats=lambda index: None,
            max_memory_allocated=lambda index: 999,
            max_memory_reserved=lambda index: 999,
        )
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = torch_stub
        try:
            with TemporaryDirectory() as tmp:
                output = Path(tmp) / "resource_metrics.json"
                with module.ResourceMonitor(
                    output_path=output,
                    method="CATCH",
                    dataset="MSL",
                    seed=42,
                    stage="inference",
                    device="cuda:0",
                    target_pid=os.getpid() + 100000,
                    sample_interval=0.01,
                ):
                    time.sleep(0.01)
                gpu = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]["gpu"]
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch

        self.assertIsNone(gpu["peak_allocated_bytes"])
        self.assertIsNone(gpu["peak_reserved_bytes"])
        self.assertIn("in-process", gpu["reason"])

    def test_monitor_storage_failure_does_not_interrupt_workload(self) -> None:
        module = load_resource_monitor_module()

        class BrokenStore:
            def upsert(self, attempt):
                raise OSError("disk unavailable")

        with TemporaryDirectory() as tmp:
            monitor = module.ResourceMonitor(
                output_path=Path(tmp) / "resource_metrics.json",
                method="SCAR",
                dataset="MSL",
                seed=42,
                stage="stage_a",
                device="cpu",
                sample_interval=0.01,
            )
            monitor._store = BrokenStore()
            with monitor:
                monitor.set_workload(points=10)
            self.assertEqual(monitor._attempt["status"]["state"], "completed")
            self.assertTrue(
                any("disk unavailable" in warning for warning in monitor._attempt["warnings"])
            )

    def test_cpu_time_accumulates_for_children_that_exit_before_stage_end(self) -> None:
        module = load_resource_monitor_module()
        target_pid = os.getpid() + 200000

        class FakeProcess:
            def __init__(self, fake_module, pid, sample_index):
                self.fake_module = fake_module
                self.pid = pid
                self.sample_index = sample_index

            def children(self, recursive=True):
                if self.pid == target_pid and self.sample_index == 1:
                    return [FakeProcess(self.fake_module, target_pid + 1, self.sample_index)]
                return []

            def memory_info(self):
                return types.SimpleNamespace(rss=100)

            def cpu_times(self):
                if self.pid == target_pid:
                    values = [(1.0, 1.0), (2.0, 1.5), (3.0, 2.0)]
                    user, system = values[min(self.sample_index, len(values) - 1)]
                else:
                    user, system = 0.5, 0.2
                return types.SimpleNamespace(user=user, system=system)

        fake_psutil = types.ModuleType("psutil")
        fake_psutil.__version__ = "test"
        fake_psutil.sample_index = 0

        def process_factory(pid):
            sample_index = fake_psutil.sample_index
            fake_psutil.sample_index += 1
            return FakeProcess(fake_psutil, pid, sample_index)

        fake_psutil.Process = process_factory
        previous_psutil = sys.modules.get("psutil")
        sys.modules["psutil"] = fake_psutil
        try:
            with TemporaryDirectory() as tmp:
                output = Path(tmp) / "resource_metrics.json"
                with module.ResourceMonitor(
                    output_path=output,
                    method="CATCH",
                    dataset="MSL",
                    seed=42,
                    stage="train",
                    device="cpu",
                    target_pid=target_pid,
                    sample_interval=100.0,
                ) as monitor:
                    monitor._sample_once()
                timing = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]["timing"]
        finally:
            if previous_psutil is None:
                sys.modules.pop("psutil", None)
            else:
                sys.modules["psutil"] = previous_psutil

        self.assertAlmostEqual(timing["cpu_user_seconds"], 3.5)
        self.assertAlmostEqual(timing["cpu_system_seconds"], 2.2)

    def test_corrupt_metrics_file_is_not_overwritten(self) -> None:
        module = load_resource_monitor_module()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            corrupt_text = "{not valid json"
            output.write_text(corrupt_text, encoding="utf-8")
            monitor = module.ResourceMonitor(
                output_path=output,
                method="SCAR",
                dataset="MSL",
                seed=42,
                stage="test",
                device="cpu",
                sample_interval=0.01,
            )
            with monitor:
                pass
            self.assertEqual(output.read_text(encoding="utf-8"), corrupt_text)
            self.assertTrue(
                any("invalid existing metrics" in warning for warning in monitor._attempt["warnings"])
            )

    def test_sampler_shutdown_timeout_persists_failed_terminal_attempt(self) -> None:
        module = load_resource_monitor_module()

        class StuckThread:
            def join(self, timeout=None):
                return None

            def is_alive(self):
                return True

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            monitor = module.ResourceMonitor(
                output_path=output,
                method="SCAR",
                dataset="MSL",
                seed=42,
                stage="test",
                device="cpu",
                sample_interval=0.01,
                sampling_enabled=False,
            )
            monitor._sampler_thread = StuckThread()
            monitor._started_perf = time.perf_counter()
            monitor._attempt["timing"]["started_at"] = "test"
            monitor.__exit__(None, None, None)
            attempt = json.loads(output.read_text(encoding="utf-8"))["attempts"][0]
            self.assertEqual(attempt["status"]["state"], "failed")
            self.assertTrue(attempt["status"]["excluded_from_summary"])
            self.assertIn("sampler thread did not stop", attempt["status"]["reason"])

    def test_failed_and_skipped_attempts_are_persisted_without_fake_gpu_values(self) -> None:
        module = load_resource_monitor_module()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "resource_metrics.json"
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with module.ResourceMonitor(
                    output_path=output,
                    method="SCAR",
                    dataset="MSL",
                    seed=42,
                    stage="test",
                    device="cpu",
                    invocation_id="failed-run",
                    sample_interval=0.01,
                ):
                    raise RuntimeError("boom")

            with module.ResourceMonitor(
                output_path=output,
                method="SCAR",
                dataset="MSL",
                seed=42,
                stage="stage_a",
                device="cpu",
                invocation_id="skipped-run",
                sample_interval=0.01,
            ) as monitor:
                monitor.mark_skipped("checkpoint already complete")

            payload = json.loads(output.read_text(encoding="utf-8"))
            failed, skipped = payload["attempts"]
            self.assertEqual(failed["status"]["state"], "failed")
            self.assertEqual(failed["status"]["exception_type"], "RuntimeError")
            self.assertEqual(skipped["status"]["state"], "skipped")
            self.assertTrue(skipped["status"]["excluded_from_summary"])
            self.assertIsNone(skipped["gpu"]["peak_allocated_bytes"])
            self.assertTrue(skipped["gpu"]["reason"])
            self.assertEqual(payload["summary"]["invocation_id"], "skipped-run")
            self.assertEqual(payload["summary"]["end_to_end_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
