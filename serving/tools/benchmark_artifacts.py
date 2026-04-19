from __future__ import annotations

import argparse
from pathlib import Path

from app.backends.onnx_backend import OnnxBackend
from app.backends.sklearn_backend import SklearnBackend
from app.config import get_settings
from app.feature_adapter import build_feature_frame
from app.schemas import PredictRequest
from tools.process_sampler import ProcessTreeSampler
from tools.common import percentile, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct artifact benchmark without HTTP serving.")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 ** 2), 4)


def bench_backend(name: str, backend, frame, artifact_path: Path, trials: int, warmup: int) -> dict:
    import time

    for _ in range(warmup):
        backend.predict(frame)

    sampler = ProcessTreeSampler()
    sampler.start()

    latencies_ms = []
    for _ in range(trials):
        start = time.perf_counter()
        backend.predict(frame)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    resource = sampler.stop()
    return {
        "study": "artifact_microbenchmark",
        "option": name,
        "backend_kind": backend.kind,
        "artifact_path": str(artifact_path),
        "artifact_size_mb": file_size_mb(artifact_path),
        "concurrency_tested": 1,
        "batch_size": 1,
        "p50_latency_ms": round(percentile(latencies_ms, 50), 4),
        "p95_latency_ms": round(percentile(latencies_ms, 95), 4),
        "throughput_rps": round(trials / (sum(latencies_ms) / 1000.0), 4),
        "error_rate": 0.0,
        "cpu_pct_avg": resource.cpu_pct_avg,
        "cpu_pct_max": resource.cpu_pct_max,
        "memory_mb_avg": resource.rss_mb_avg,
        "memory_mb_max": resource.rss_mb_max,
    }


def main() -> None:
    args = parse_args()
    settings = get_settings()
    root = Path(__file__).resolve().parents[1]

    source_path = Path(settings.source_model_path)
    onnx_path = root / "models/optimized/v2_tfidf_linearsvc_model.onnx"
    quant_path = root / "models/optimized/v2_tfidf_linearsvc_model.dynamic_quant.onnx"

    frame = build_feature_frame(
        [
            PredictRequest(
                transaction_description="STARBUCKS STORE 1458 NEW YORK NY",
                country="US",
                currency="USD",
            )
        ]
    )

    rows = [
        bench_backend(
            "baseline_direct",
            SklearnBackend(str(source_path)),
            frame,
            source_path,
            args.trials,
            args.warmup,
        ),
        bench_backend(
            "onnx_direct",
            OnnxBackend(str(onnx_path), str(source_path)),
            frame,
            onnx_path,
            args.trials,
            args.warmup,
        ),
        bench_backend(
            "onnx_dynamic_quant_direct",
            OnnxBackend(str(quant_path), str(source_path)),
            frame,
            quant_path,
            args.trials,
            args.warmup,
        ),
    ]
    write_json(Path(args.output_json), rows)
    print(Path(args.output_json).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
