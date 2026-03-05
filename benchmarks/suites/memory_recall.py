"""Memory recall benchmark: seed N facts, M distractor turns, query."""

import random
import time
from typing import List

from ..metrics import BenchmarkMetrics, LatencyTracker, compute_exact_match, compute_llm_match, compute_latency


# Fact templates with matching query templates
_FACT_TEMPLATES = [
    ("{entity} likes {attr}", "What does {entity} like?"),
    ("{entity} lives in {attr}", "Where does {entity} live?"),
    ("{entity} works at {attr}", "Where does {entity} work?"),
    ("{entity} drives a {attr}", "What does {entity} drive?"),
    ("{entity} owns a {attr}", "What does {entity} own?"),
]

_ENTITIES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona",
    "George", "Hannah", "Ivan", "Julia", "Kevin", "Laura",
    "Marcus", "Nina", "Oscar", "Petra", "Quinn", "Rachel",
]

_ATTRIBUTES = [
    "red", "blue", "green", "Portland", "Seattle", "Chicago",
    "Google", "Apple", "Toyota", "Tesla", "cat", "guitar",
]


class MemoryRecallSuite:
    """Benchmark: seed k facts, add distractors, then query."""

    def __init__(self, k_values: List[int] = None, n_distractors: int = 5, n_queries: int = 20):
        self.k_values = k_values or [10, 25, 50, 100]
        self.n_distractors = n_distractors
        self.n_queries = n_queries

    def run(self, baseline) -> BenchmarkMetrics:
        """Run the benchmark on a baseline.

        baseline must implement:
          - reset()
          - teach(text: str)
          - query(text: str) -> str
        """
        rng = random.Random(42)
        all_predictions = []
        all_targets = []
        all_latencies = []

        for k in self.k_values:
            baseline.reset()

            # Generate facts
            entities = rng.sample(_ENTITIES, min(k, len(_ENTITIES)))
            while len(entities) < k:
                entities.append(f"Person{len(entities)}")

            facts = {}
            for ent in entities:
                fact_template, query_template = rng.choice(_FACT_TEMPLATES)
                attr = rng.choice(_ATTRIBUTES)
                fact_text = fact_template.format(entity=ent, attr=attr)
                query_text = query_template.format(entity=ent)
                facts[ent] = (attr, fact_text, query_text)
                baseline.teach(fact_text)

            # Distractor turns
            for _ in range(self.n_distractors):
                baseline.teach(f"The weather is {rng.choice(['sunny', 'rainy', 'cloudy'])} today")

            # Query phase
            query_entities = rng.sample(list(facts.keys()), min(self.n_queries, len(facts)))
            for ent in query_entities:
                target_attr, _, query_text = facts[ent]

                with LatencyTracker() as lt:
                    response = baseline.query(query_text)

                all_predictions.append(response)
                all_targets.append(target_attr)
                all_latencies.append(lt.elapsed_ms)

        exact_match = compute_exact_match(all_predictions, all_targets)
        llm_match = compute_llm_match(all_predictions, all_targets)  # D-229
        p50, p95 = compute_latency(all_latencies)

        return BenchmarkMetrics(
            exact_match=exact_match,
            llm_match=llm_match,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            total_queries=len(all_predictions),
            correct=int(llm_match * len(all_predictions)),
        )
