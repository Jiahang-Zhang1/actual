#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
ENV_FILE = REPO_ROOT / ".env"
SERVICE_CONTAINER_NAME = "actualbudget-serving-app"
DEFAULT_HOST_URL = "http://127.0.0.1:8000"
TOOLING_BASE_URL = "http://serve:8000"
DEFAULT_ENV_EXAMPLE_CONTENT = """HOST_PORT=8000
SERVICE_PORT=8000

BACKEND_KIND=onnx_dynamic_quant
MODEL_PATH=/workspace/models/optimized/v2_tfidf_linearsvc_model.dynamic_quant.onnx
SOURCE_MODEL_PATH=/workspace/models/source/v2_tfidf_linearsvc_model.joblib

MODEL_VERSION=v2_tfidf_linearsvc
TOP_K=3
WEB_CONCURRENCY=2
LOG_LEVEL=info

SERVICE_CPUS=2.0
SERVICE_MEM_LIMIT=3g
"""


def ensure_env_file() -> None:
    if not ENV_EXAMPLE.exists():
        ENV_EXAMPLE.write_text(DEFAULT_ENV_EXAMPLE_CONTENT, encoding="utf-8")
        print("Created .env.example from built-in defaults")

    if not ENV_FILE.exists():
        shutil.copy2(ENV_EXAMPLE, ENV_FILE)
        print(f"Created {ENV_FILE.name} from .env.example")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except PermissionError:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


def run(cmd: list[str], env: dict | None = None, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    env_combined = os.environ.copy()
    if env:
        env_combined.update({k: str(v) for k, v in env.items()})
    print("$", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env_combined,
        check=check,
        text=True,
        capture_output=capture,
    )


def compose_cmd(*parts: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_FILE), *parts]


def docker_available() -> bool:
    try:
        run(["docker", "--version"], capture=True)
        run(["docker", "compose", "version"], capture=True)
        return True
    except Exception:
        return False


def require_docker() -> None:
    if docker_available():
        return
    raise SystemExit(
        "Docker is required but was not found. On a fresh Chameleon instance, run: \
"
        "  sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2\n"
        "  sudo systemctl enable --now docker\n"
        "  sudo usermod -aG docker $USER && newgrp docker\n"
    )


def backend_default_model_path(variant: str) -> str:
    if variant == "baseline":
        return "/workspace/models/source/v2_tfidf_linearsvc_model.joblib"
    if variant == "onnx":
        return "/workspace/models/optimized/v2_tfidf_linearsvc_model.onnx"
    if variant == "onnx_dynamic_quant":
        return "/workspace/models/optimized/v2_tfidf_linearsvc_model.dynamic_quant.onnx"
    raise ValueError(f"Unsupported variant: {variant}")


def build() -> None:
    ensure_env_file()
    require_docker()
    run(compose_cmd("build"))


def down() -> None:
    ensure_env_file()
    require_docker()
    run(compose_cmd("down", "--remove-orphans"), check=False)


def logs() -> None:
    ensure_env_file()
    require_docker()
    run(compose_cmd("logs", "-f", "serve"))


def monitor_up() -> None:
    ensure_env_file()
    require_docker()
    run(compose_cmd("up", "-d", "serve", "prometheus", "grafana"))


def monitor_down() -> None:
    ensure_env_file()
    require_docker()
    run(compose_cmd("stop", "prometheus", "grafana"), check=False)


def http_get_json(url: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def wait_until_ready(base_url: str, timeout_s: float = 120.0) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        try:
            payload = http_get_json(f"{base_url.rstrip('/')}/readyz", timeout=3.0)
            if payload.get("ready"):
                return round(time.perf_counter() - start, 3)
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Service did not become ready within {timeout_s} seconds.")


def smoke(base_url: str = DEFAULT_HOST_URL) -> None:
    payload = (REPO_ROOT / "artifacts/examples/input_sample.json").read_text(encoding="utf-8").encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as response:
        body = response.read().decode("utf-8")
    print(body)


def prepare(force: bool = False) -> None:
    build()
    args = ["run", "--rm", "tooling", "-m", "tools.prepare_artifacts"]
    if force:
        args.append("--force")
    run(compose_cmd(*args))


def up(variant: str, workers: int, cpus: str, mem: str, host_port: int, rebuild: bool = False) -> float:
    ensure_env_file()
    require_docker()
    model_path = backend_default_model_path(variant)
    env = {
        "BACKEND_KIND": variant,
        "MODEL_PATH": model_path,
        "WEB_CONCURRENCY": str(workers),
        "SERVICE_CPUS": str(cpus),
        "SERVICE_MEM_LIMIT": str(mem),
        "HOST_PORT": str(host_port),
    }
    down()
    cmd = compose_cmd("up", "-d")
    if rebuild:
        cmd.append("--build")
    cmd.append("serve")
    run(cmd, env=env)
    ready_after_s = wait_until_ready(f"http://127.0.0.1:{host_port}")
    print(f"Service ready after {ready_after_s}s")
    return ready_after_s


def docker_stats_once(container_name: str) -> dict | None:
    proc = run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}", container_name],
        capture=True,
        check=False,
    )
    raw = (proc.stdout or "").strip().splitlines()
    if not raw:
        return None
    try:
        row = json.loads(raw[0])
    except json.JSONDecodeError:
        return None

    def parse_percent(text: str | None) -> float | None:
        if not text:
            return None
        text = text.strip().replace("%", "")
        try:
            return float(text)
        except ValueError:
            return None

    def parse_memory_mb(text: str | None) -> float | None:
        if not text:
            return None
        current = text.split("/")[0].strip()
        match = re.match(r"([0-9.]+)\s*([KMG]i?B|[KMG]B)", current)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2)
        multipliers = {
            "KiB": 1 / 1024.0,
            "KB": 1 / 1024.0,
            "MiB": 1.0,
            "MB": 1.0,
            "GiB": 1024.0,
            "GB": 1024.0,
        }
        return round(value * multipliers[unit], 4)

    return {
        "ts": time.time(),
        "cpu_pct": parse_percent(row.get("CPUPerc")),
        "memory_mb": parse_memory_mb(row.get("MemUsage")),
        "memory_pct": parse_percent(row.get("MemPerc")),
        "container": row.get("Name") or container_name,
    }


class DockerStatsMonitor:
    def __init__(self, container_name: str, interval_s: float = 0.5):
        self.container_name = container_name
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = docker_stats_once(self.container_name)
            if sample is not None:
                self.samples.append(sample)
            time.sleep(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if not self.samples:
            return {
                "cpu_pct_avg": None,
                "cpu_pct_max": None,
                "memory_mb_avg": None,
                "memory_mb_max": None,
                "sample_count": 0,
            }
        cpu_values = [x["cpu_pct"] for x in self.samples if x["cpu_pct"] is not None]
        mem_values = [x["memory_mb"] for x in self.samples if x["memory_mb"] is not None]
        return {
            "cpu_pct_avg": round(sum(cpu_values) / len(cpu_values), 4) if cpu_values else None,
            "cpu_pct_max": round(max(cpu_values), 4) if cpu_values else None,
            "memory_mb_avg": round(sum(mem_values) / len(mem_values), 4) if mem_values else None,
            "memory_mb_max": round(max(mem_values), 4) if mem_values else None,
            "sample_count": len(self.samples),
        }


def tooling(module: str, *module_args: str, env: dict | None = None) -> None:
    ensure_env_file()
    require_docker()
    run(compose_cmd("run", "--rm", "tooling", "-m", module, *module_args), env=env)


def run_http_case(
    study: str,
    option: str,
    variant: str,
    workers: int,
    cpus: str,
    mem: str,
    concurrency: int,
    endpoint: str = "/predict",
    batch_size: int = 1,
    requests_count: int = 80,
) -> None:
    output_path = REPO_ROOT / "results/summary/_temp_http.json"
    monitor = DockerStatsMonitor(SERVICE_CONTAINER_NAME)
    monitor.start()
    tooling(
        "tools.benchmark_http",
        "--base-url", TOOLING_BASE_URL,
        "--endpoint", endpoint,
        "--request-json", "/workspace/artifacts/examples/input_sample.json",
        "--concurrency", str(concurrency),
        "--requests", str(requests_count),
        "--batch-size", str(batch_size),
        "--output-json", f"/workspace/{output_path.relative_to(REPO_ROOT)}",
    )
    resource = monitor.stop()
    row = read_json(output_path)
    row.update(
        {
            "study": study,
            "option": option,
            "workers": workers,
            "service_cpus_limit": str(cpus),
            "service_mem_limit": str(mem),
            "batch_size": batch_size,
            "arrival_distribution": None,
            **resource,
        }
    )
    append_jsonl(REPO_ROOT / f"results/raw/{study}.jsonl", row)
    output_path.unlink(missing_ok=True)


def run_arrival_case(
    study: str,
    option: str,
    workers: int,
    cpus: str,
    mem: str,
    distribution: str,
    request_rate: float,
    duration: float = 15.0,
) -> None:
    output_path = REPO_ROOT / "results/summary/_temp_arrivals.json"
    monitor = DockerStatsMonitor(SERVICE_CONTAINER_NAME)
    monitor.start()
    tooling(
        "tools.benchmark_arrivals",
        "--base-url", TOOLING_BASE_URL,
        "--endpoint", "/predict",
        "--request-json", "/workspace/artifacts/examples/input_sample.json",
        "--request-rate", str(request_rate),
        "--duration", str(duration),
        "--distribution", distribution,
        "--output-json", f"/workspace/{output_path.relative_to(REPO_ROOT)}",
    )
    resource = monitor.stop()
    row = read_json(output_path)
    row.update(
        {
            "study": study,
            "option": option,
            "workers": workers,
            "service_cpus_limit": str(cpus),
            "service_mem_limit": str(mem),
            "concurrency_tested": None,
            "batch_size": 1,
            "arrival_distribution": distribution,
            **resource,
        }
    )
    append_jsonl(REPO_ROOT / f"results/raw/{study}.jsonl", row)
    output_path.unlink(missing_ok=True)


def run_artifact_benchmark() -> None:
    output_path = REPO_ROOT / "results/summary/_temp_artifacts.json"
    tooling(
        "tools.benchmark_artifacts",
        "--output-json", f"/workspace/{output_path.relative_to(REPO_ROOT)}",
    )
    rows = read_json(output_path)
    for row in rows:
        append_jsonl(REPO_ROOT / "results/raw/model_artifacts.jsonl", row)
    output_path.unlink(missing_ok=True)


def bench_model(ensure_prepared: bool = True) -> None:
    if ensure_prepared:
        prepare(force=False)
    for variant in ["baseline", "onnx", "onnx_dynamic_quant"]:
        up(variant=variant, workers=1, cpus="2.0", mem="3g", host_port=8000, rebuild=False)
        for concurrency in [1, 4, 8]:
            run_http_case(
                study="model_http",
                option=f"{variant}_w1_c{concurrency}",
                variant=variant,
                workers=1,
                cpus="2.0",
                mem="3g",
                concurrency=concurrency,
                endpoint="/predict",
                batch_size=1,
                requests_count=80,
            )
    run_artifact_benchmark()


def bench_system(ensure_prepared: bool = True) -> None:
    if ensure_prepared:
        prepare(force=False)
    variant = "onnx_dynamic_quant"

    # worker sweep
    for workers in [1, 2, 4]:
        up(variant=variant, workers=workers, cpus="2.0", mem="3g", host_port=8000, rebuild=False)
        run_http_case(
            study="system_workers",
            option=f"{variant}_workers_{workers}",
            variant=variant,
            workers=workers,
            cpus="2.0",
            mem="3g",
            concurrency=8,
            endpoint="/predict",
            batch_size=1,
            requests_count=100,
        )

    # concurrency sweep
    up(variant=variant, workers=2, cpus="2.0", mem="3g", host_port=8000, rebuild=False)
    for concurrency in [1, 4, 8, 16]:
        run_http_case(
            study="system_concurrency",
            option=f"{variant}_w2_concurrency_{concurrency}",
            variant=variant,
            workers=2,
            cpus="2.0",
            mem="3g",
            concurrency=concurrency,
            endpoint="/predict",
            batch_size=1,
            requests_count=100,
        )

    # batch sweep
    up(variant=variant, workers=2, cpus="2.0", mem="3g", host_port=8000, rebuild=False)
    for batch_size in [1, 4, 8, 16]:
        endpoint = "/predict" if batch_size == 1 else "/predict_batch"
        run_http_case(
            study="system_batch",
            option=f"{variant}_batch_{batch_size}",
            variant=variant,
            workers=2,
            cpus="2.0",
            mem="3g",
            concurrency=4,
            endpoint=endpoint,
            batch_size=batch_size,
            requests_count=80,
        )

    # arrival patterns
    up(variant=variant, workers=2, cpus="2.0", mem="3g", host_port=8000, rebuild=False)
    for distribution in ["constant", "poisson"]:
        for request_rate in [10, 20, 40]:
            run_arrival_case(
                study="system_arrivals",
                option=f"{variant}_{distribution}_{int(request_rate)}rps",
                workers=2,
                cpus="2.0",
                mem="3g",
                distribution=distribution,
                request_rate=request_rate,
                duration=12.0,
            )


def bench_infra(ensure_prepared: bool = True) -> None:
    if ensure_prepared:
        prepare(force=False)
    variant = "onnx_dynamic_quant"
    profiles = [
        ("small", "1.0", "1g", 1),
        ("recommended", "2.0", "2g", 2),
        ("headroom", "2.0", "3g", 2),
    ]
    for name, cpus, mem, workers in profiles:
        ready_after_s = up(variant=variant, workers=workers, cpus=cpus, mem=mem, host_port=8000, rebuild=False)
        run_http_case(
            study="infra_profiles",
            option=f"{variant}_{name}",
            variant=variant,
            workers=workers,
            cpus=cpus,
            mem=mem,
            concurrency=8,
            endpoint="/predict",
            batch_size=1,
            requests_count=120,
        )
        # patch last row with readiness
        target = REPO_ROOT / "results/raw/infra_profiles.jsonl"
        rows = target.read_text(encoding="utf-8").splitlines()
        if rows:
            last = json.loads(rows[-1])
            last["ready_after_s"] = ready_after_s
            rows[-1] = json.dumps(last)
            target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def package() -> None:
    tooling("tools.package_submission")


def full() -> None:
    prepare(force=False)
    bench_model(ensure_prepared=False)
    bench_system(ensure_prepared=False)
    bench_infra(ensure_prepared=False)
    package()


def doctor() -> None:
    ensure_env_file()
    checks = {
        "docker_available": docker_available(),
        "compose_file_exists": COMPOSE_FILE.exists(),
        "source_model_exists": (REPO_ROOT / "models/source/v2_tfidf_linearsvc_model.joblib").exists(),
        "input_sample_exists": (REPO_ROOT / "artifacts/examples/input_sample.json").exists(),
    }
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-folder orchestrator for ActualBudget serving.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor")
    sub.add_parser("build")
    sub.add_parser("monitor-up")
    sub.add_parser("monitor-down")

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--force", action="store_true")

    p_up = sub.add_parser("up")
    p_up.add_argument("--variant", choices=["baseline", "onnx", "onnx_dynamic_quant"], default="onnx_dynamic_quant")
    p_up.add_argument("--workers", type=int, default=2)
    p_up.add_argument("--cpus", default="2.0")
    p_up.add_argument("--mem", default="3g")
    p_up.add_argument("--host-port", type=int, default=8000)
    p_up.add_argument("--rebuild", action="store_true")

    sub.add_parser("down")
    sub.add_parser("logs")
    sub.add_parser("smoke")
    sub.add_parser("bench-model")
    sub.add_parser("bench-system")
    sub.add_parser("bench-infra")
    sub.add_parser("bench-all")
    sub.add_parser("package")
    sub.add_parser("full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == "build":
        build()
    elif args.command == "monitor-up":
        monitor_up()
    elif args.command == "monitor-down":
        monitor_down()
    elif args.command == "prepare":
        prepare(force=args.force)
    elif args.command == "up":
        up(args.variant, args.workers, args.cpus, args.mem, args.host_port, rebuild=args.rebuild)
    elif args.command == "down":
        down()
    elif args.command == "logs":
        logs()
    elif args.command == "smoke":
        smoke()
    elif args.command == "bench-model":
        bench_model()
    elif args.command == "bench-system":
        bench_system()
    elif args.command == "bench-infra":
        bench_infra()
    elif args.command == "bench-all":
        prepare(force=False)
        bench_model(ensure_prepared=False)
        bench_system(ensure_prepared=False)
        bench_infra(ensure_prepared=False)
    elif args.command == "package":
        package()
    elif args.command == "full":
        full()
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
