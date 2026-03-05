"""Learning transfer benchmark: teach rules/concepts, test application.

Ported from nexus-1's learning_transfer suite, adapted to
nexus-2's reset/teach/query baseline interface.
"""

import random
from typing import Dict, List

from ..metrics import BenchmarkMetrics, LatencyTracker, compute_latency


_TEACHING_DOCS = [
    "Rule A: CODE(x) shifts each letter forward by one and uppercases output. Example: CODE(cat)=DBU.",
    "Rule B: Project Kappa is owned by Lina. Lina is located in Seoul.",
    "Rule C: Emergency protocol token is channel-7.",
]

_APPLICATION_TASKS = [
    {"id": "LT01", "q": "Using Rule A, what is CODE(map)?", "keywords": ["NBQ"], "category": "rule_apply"},
    {"id": "LT02", "q": "Which city is the owner of Project Kappa located in?", "keywords": ["seoul"], "category": "inference"},
    {"id": "LT03", "q": "What is the emergency protocol token?", "keywords": ["channel-7"], "category": "retention"},
]


class LearningTransferSuite:
    """Benchmark: teach rules via docs, test application in novel contexts."""

    def __init__(self, task_repetitions: int = 1, **kwargs):
        self.task_repetitions = task_repetitions

    def run(self, baseline) -> BenchmarkMetrics:
        reps = self.task_repetitions
        all_predictions: List[str] = []
        all_targets: List[str] = []
        all_latencies: List[float] = []
        category_results: Dict[str, List[bool]] = {}

        baseline.reset()

        # Teach phase
        for doc in _TEACHING_DOCS:
            baseline.teach(doc)

        # Apply phase
        for rep in range(reps):
            for task in _APPLICATION_TASKS:
                cat = task["category"]
                target = task["keywords"][0]

                with LatencyTracker() as lt:
                    response = baseline.query(task["q"])

                all_predictions.append(response)
                all_targets.append(target)
                all_latencies.append(lt.elapsed_ms)

                hit = target.lower() in response.lower()
                category_results.setdefault(cat, []).append(hit)

        correct = sum(
            1 for p, t in zip(all_predictions, all_targets)
            if t.lower() in p.lower()
        )
        accuracy = correct / len(all_predictions) if all_predictions else 0.0
        p50, p95 = compute_latency(all_latencies)

        return BenchmarkMetrics(
            exact_match=accuracy,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            total_queries=len(all_predictions),
            correct=correct,
        )
