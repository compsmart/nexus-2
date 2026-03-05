"""Benchmark orchestrator: runs suites x baselines, collects metrics."""

import time
from typing import Dict, List, Optional

from .metrics import BenchmarkMetrics, LatencyTracker


class BenchmarkRunner:
    """Orchestrates benchmark suites across multiple baselines."""

    def __init__(self):
        self._suites: Dict[str, object] = {}
        self._baselines: Dict[str, object] = {}

    def register_suite(self, name: str, suite):
        """Register a benchmark suite."""
        self._suites[name] = suite

    def register_baseline(self, name: str, baseline):
        """Register a baseline system."""
        self._baselines[name] = baseline

    def run(
        self,
        suite_names: Optional[List[str]] = None,
        baseline_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, BenchmarkMetrics]]:
        """Run selected suites against selected baselines.

        Returns: {suite_name: {baseline_name: BenchmarkMetrics}}
        """
        suites = suite_names or list(self._suites.keys())
        baselines = baseline_names or list(self._baselines.keys())

        results = {}
        total = len(suites) * len(baselines)
        step = 0

        for suite_name in suites:
            suite = self._suites.get(suite_name)
            if suite is None:
                continue
            results[suite_name] = {}

            for baseline_name in baselines:
                step += 1
                baseline = self._baselines.get(baseline_name)
                if baseline is None:
                    continue

                print(
                    f"  [{step}/{total}] {suite_name} x {baseline_name}...",
                    flush=True,
                )

                try:
                    metrics = suite.run(baseline)
                    results[suite_name][baseline_name] = metrics
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    results[suite_name][baseline_name] = BenchmarkMetrics()

        return results

    def format_results(self, results: Dict[str, Dict[str, BenchmarkMetrics]]) -> str:
        """Format results as a markdown table."""
        lines = []

        for suite_name, baselines in results.items():
            lines.append(f"\n## {suite_name}\n")
            # Header
            bl_names = list(baselines.keys())
            lines.append("| Metric | " + " | ".join(bl_names) + " |")
            lines.append("|" + "---|" * (len(bl_names) + 1))

            # Rows
            lines.append(
                "| Exact Match | "
                + " | ".join(f"{baselines[b].exact_match:.3f}" for b in bl_names)
                + " |"
            )
            lines.append(
                "| LLM Match (D-229) | "
                + " | ".join(f"{baselines[b].llm_match:.3f}" for b in bl_names)
                + " |"
            )
            lines.append(
                "| Latency p50 (ms) | "
                + " | ".join(f"{baselines[b].latency_p50_ms:.1f}" for b in bl_names)
                + " |"
            )
            lines.append(
                "| Latency p95 (ms) | "
                + " | ".join(f"{baselines[b].latency_p95_ms:.1f}" for b in bl_names)
                + " |"
            )

            # Hop success rates
            all_hops = set()
            for b in bl_names:
                all_hops.update(baselines[b].hop_success_rate.keys())
            for hop in sorted(all_hops):
                lines.append(
                    f"| {hop}-hop accuracy | "
                    + " | ".join(
                        f"{baselines[b].hop_success_rate.get(hop, 0):.3f}"
                        for b in bl_names
                    )
                    + " |"
                )

        return "\n".join(lines)
