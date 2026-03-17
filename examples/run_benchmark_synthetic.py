"""Run a synthetic benchmark comparing classical and deep configurations."""

from __future__ import annotations

from pathlib import Path

from clustro.benchmark.runner import run_classical_vs_deep_benchmark


def main() -> None:
    root = Path(__file__).resolve().parent / "generated" / "benchmark_synthetic"
    summary = run_classical_vs_deep_benchmark(root)
    print(summary.to_string(index=False))
    print(f"benchmark results: {root}")


if __name__ == "__main__":
    main()
