"""CLI entry point for NEXUS-2 benchmarks.

Usage:
    python benchmark.py --suite all                         # Full run
    python benchmark.py --suite multihop --baselines nexus2,rag  # Targeted
    python benchmark.py --suite memory_recall --device cuda  # GPU run
"""

import argparse
import sys

from benchmarks.runner import BenchmarkRunner
from benchmarks.suites.memory_recall import MemoryRecallSuite
from benchmarks.suites.multihop_chain import MultihopChainSuite
from benchmarks.suites.scalability import ScalabilitySuite
from benchmarks.suites.vs_rag import VsRagSuite


def build_runner(suite_names, baseline_names, device):
    """Build configured benchmark runner."""
    runner = BenchmarkRunner()

    # Register suites
    all_suites = {
        "memory_recall": MemoryRecallSuite(),
        "multihop": MultihopChainSuite(),
        "scalability": ScalabilitySuite(),
        "vs_rag": VsRagSuite(),
    }

    for name in (suite_names or all_suites.keys()):
        if name in all_suites:
            runner.register_suite(name, all_suites[name])

    # Register baselines
    all_baselines = {}

    if "nexus2" in baseline_names or not baseline_names:
        from benchmarks.baselines.nexus2_baseline import Nexus2Baseline
        all_baselines["nexus2"] = Nexus2Baseline(device=device)

    if "rag" in baseline_names or not baseline_names:
        from benchmarks.baselines.rag_baseline import RagBaseline
        all_baselines["rag"] = RagBaseline(device=device)

    if "llm_only" in baseline_names or "phi_only" in baseline_names or not baseline_names:
        from benchmarks.baselines.phi_only_baseline import LLMOnlyBaseline
        all_baselines["llm_only"] = LLMOnlyBaseline(device=device)

    if "nexus1" in baseline_names:
        from benchmarks.baselines.nexus1_baseline import Nexus1Baseline
        all_baselines["nexus1"] = Nexus1Baseline(device=device)

    for name, bl in all_baselines.items():
        runner.register_baseline(name, bl)

    return runner


def main():
    parser = argparse.ArgumentParser(description="NEXUS-2 Benchmarks")
    parser.add_argument(
        "--suite",
        default="all",
        help="Comma-separated suite names or 'all'",
    )
    parser.add_argument(
        "--baselines",
        default="",
        help="Comma-separated baseline names (default: all)",
    )
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda")
    args = parser.parse_args()

    suite_names = None if args.suite == "all" else args.suite.split(",")
    baseline_names = [b.strip() for b in args.baselines.split(",") if b.strip()]

    print("=" * 60, flush=True)
    print("NEXUS-2 Benchmark Suite", flush=True)
    print("=" * 60, flush=True)
    print(f"  Suites: {args.suite}", flush=True)
    print(f"  Baselines: {args.baselines or 'all'}", flush=True)
    print(f"  Device: {args.device}", flush=True)
    print()

    runner = build_runner(suite_names, baseline_names, args.device)
    results = runner.run()
    report = runner.format_results(results)

    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(report, flush=True)

    # Save report
    with open("benchmark_results.md", "w") as f:
        f.write("# NEXUS-2 Benchmark Results\n\n")
        f.write(report)
    print("\nResults saved to benchmark_results.md", flush=True)


if __name__ == "__main__":
    main()
