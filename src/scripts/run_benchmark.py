"""
Профилировщик: сравнивает Итерацию 1 (dota2_flat, exact) и Итерацию 2
(dota2_quantized, INT8 + HNSW) по латентности, RPS и пиковому потреблению
RAM процессом Qdrant.

Запуск:
    python scripts/run_benchmark.py --requests 100
"""

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import psutil

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.adapters.model_adapter import ModelAdapter  # noqa: E402
from src.core.config import settings  # noqa: E402
from src.database.operations import search as vector_search  # noqa: E402

SAMPLE_QUERIES = [
    "лучшая игра для игры с друзьями",
    "слишком много токсичности в чате",
    "затягивает на сотни часов",
    "баланс героев сломан после патча",
    "лучшая моба всех времен",
    "разработчики забросили игру",
    "нужен мощный компьютер чтобы играть",
    "тиммейты постоянно фидят",
    "любимая игра детства",
    "матчмейкинг работает ужасно долго",
]


def find_qdrant_process() -> Optional[psutil.Process]:
    """Best-effort поиск процесса Qdrant среди локальных процессов."""
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "qdrant" in name or "qdrant" in cmdline:
                return psutil.Process(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def measure_ram_mb(proc: Optional[psutil.Process]) -> Optional[float]:
    if proc is None:
        return None
    try:
        return proc.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct / 100))
    return sorted_values[idx]


def run_benchmark(
    collection_key: str,
    collection_name: str,
    exact: bool,
    n_requests: int,
    embedder: ModelAdapter,
    qdrant_proc: Optional[psutil.Process],
) -> Dict[str, float]:
    latencies_ms: List[float] = []
    peak_ram_mb = measure_ram_mb(qdrant_proc) or 0.0

    start_all = time.perf_counter()
    for i in range(n_requests):
        query_text = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        vector = embedder.encode(query_text)

        t0 = time.perf_counter()
        vector_search(
            collection_name=collection_name,
            query_vector=vector,
            top_k=10,
            exact=exact,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        current_ram = measure_ram_mb(qdrant_proc)
        if current_ram is not None:
            peak_ram_mb = max(peak_ram_mb, current_ram)

    total_time_s = time.perf_counter() - start_all
    sorted_latencies = sorted(latencies_ms)

    return {
        "collection": collection_key,
        "requests": float(n_requests),
        "total_time_s": total_time_s,
        "rps": n_requests / total_time_s if total_time_s > 0 else 0.0,
        "avg_latency_ms": statistics.mean(latencies_ms),
        "p50_latency_ms": statistics.median(latencies_ms),
        "p95_latency_ms": percentile(sorted_latencies, 95),
        "p99_latency_ms": percentile(sorted_latencies, 99),
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "peak_ram_mb": peak_ram_mb,
    }


def print_report(flat_stats: Dict[str, float], quant_stats: Dict[str, float], ram_available: bool) -> None:
    line = "=" * 72
    print(line)
    print("ОТЧЁТ БЕНЧМАРКА: dota2_flat (Baseline, exact) vs dota2_quantized (INT8)")
    print(line)
    print(f"{'Метрика':<30}{'Flat (exact)':>19}{'Quantized (INT8)':>22}")
    print("-" * 72)

    def row(label: str, key: str, fmt: str = "{:.2f}") -> None:
        v1 = fmt.format(flat_stats[key])
        v2 = fmt.format(quant_stats[key])
        print(f"{label:<30}{v1:>19}{v2:>22}")

    row("RPS (запросов/сек)", "rps")
    row("Средняя латентность, ms", "avg_latency_ms")
    row("P50 латентность, ms", "p50_latency_ms")
    row("P95 латентность, ms", "p95_latency_ms")
    row("P99 латентность, ms", "p99_latency_ms")
    row("Мин. латентность, ms", "min_latency_ms")
    row("Макс. латентность, ms", "max_latency_ms")

    if ram_available:
        row("Пиковая RAM Qdrant, MB", "peak_ram_mb")
    else:
        print(f"{'Пиковая RAM Qdrant, MB':<30}{'N/A':>19}{'N/A':>22}")
        print("  Процесс Qdrant не найден локально через psutil — вероятно,")
        print("  он изолирован в Docker (Docker Desktop VM на macOS/Windows).")
        print("  Для оценки RAM в этом случае используйте: docker stats dota2_qdrant")

    print(line)
    if quant_stats["avg_latency_ms"] > 0:
        speedup = flat_stats["avg_latency_ms"] / quant_stats["avg_latency_ms"]
        print(f"Ускорение quantized относительно flat (по средней латентности): {speedup:.2f}x")
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк поиска Qdrant: flat vs quantized")
    parser.add_argument("--requests", type=int, default=100, help="Количество запросов на коллекцию")
    args = parser.parse_args()

    embedder = ModelAdapter(
        vector_size=settings.vector_size,
        use_mock=settings.use_mock_embedder,
        model_name=settings.model_name,
    )
    qdrant_proc = find_qdrant_process()
    ram_available = qdrant_proc is not None

    print(f"Запуск бенчмарка: {args.requests} запросов на коллекцию...")
    if not ram_available:
        print("[warn] Процесс Qdrant не обнаружен локально средствами psutil.")

    flat_stats = run_benchmark(
        "dota2_flat", settings.collection_flat, True, args.requests, embedder, qdrant_proc
    )
    quant_stats = run_benchmark(
        "dota2_quantized", settings.collection_quantized, False, args.requests, embedder, qdrant_proc
    )

    print_report(flat_stats, quant_stats, ram_available)


if __name__ == "__main__":
    main()

