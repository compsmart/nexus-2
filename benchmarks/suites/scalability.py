"""Scalability benchmark: accuracy degradation k=10..500."""

import random
from typing import List

from ..metrics import BenchmarkMetrics, LatencyTracker, compute_exact_match, compute_llm_match, compute_latency

_TEMPLATES = [
    "{entity} LIKES {attr}",
    "{entity} OWNS {attr}",
]

_ATTRS = ["red", "blue", "green", "gold", "silver", "iron", "oak", "pine", "elm", "maple",
          "hawk", "wolf", "bear", "fox", "deer", "swan", "crow", "dove", "lynx", "seal"]


class ScalabilitySuite:
    """Benchmark: accuracy vs k=10..500."""

    def __init__(self, k_values: List[int] = None, n_queries: int = 20):
        self.k_values = k_values or [10, 25, 50, 100, 200, 500]
        self.n_queries = n_queries

    def run(self, baseline) -> BenchmarkMetrics:
        rng = random.Random(42)
        all_predictions = []
        all_targets = []
        all_latencies = []
        recall_by_k = {}

        for k in self.k_values:
            baseline.reset()

            # Generate k facts
            facts = {}
            for i in range(k):
                entity = f"Entity_{i:04d}"
                attr = rng.choice(_ATTRS)
                template = rng.choice(_TEMPLATES)
                text = template.format(entity=entity, attr=attr)
                facts[entity] = attr
                baseline.teach(text)

            # Query subset
            entities = list(facts.keys())
            query_ents = rng.sample(entities, min(self.n_queries, len(entities)))

            correct_k = 0
            for ent in query_ents:
                target = facts[ent]
                with LatencyTracker() as lt:
                    response = baseline.query(f"What does {ent} like or own?")
                all_predictions.append(response)
                all_targets.append(target)
                all_latencies.append(lt.elapsed_ms)
                if target.lower() in response.lower():
                    correct_k += 1

            recall_by_k[k] = correct_k / len(query_ents)

        exact_match = compute_exact_match(all_predictions, all_targets)
        llm_match = compute_llm_match(all_predictions, all_targets)  # D-229
        p50, p95 = compute_latency(all_latencies)

        return BenchmarkMetrics(
            exact_match=exact_match,
            llm_match=llm_match,
            recall_at_k=recall_by_k,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            total_queries=len(all_predictions),
        )
