"""Multi-hop chain benchmark: 2-5 hop accuracy at various k values.

THE key differentiator for NEXUS-2 vs RAG systems.
"""

import random
from typing import List

from ..metrics import (
    BenchmarkMetrics, LatencyTracker,
    compute_exact_match, compute_llm_match, compute_hop_success_rate, compute_latency,
)

_ENTITIES = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
    "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima",
    "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
    "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "Xray",
]


class MultihopChainSuite:
    """Benchmark: N-hop reasoning chains at various fact counts."""

    def __init__(
        self,
        hop_values: List[int] = None,
        k_values: List[int] = None,
        n_chains_per_config: int = 10,
    ):
        self.hop_values = hop_values or [2, 3, 4, 5]
        self.k_values = k_values or [5, 10, 50, 100]
        self.n_chains = n_chains_per_config

    def run(self, baseline) -> BenchmarkMetrics:
        """Run multi-hop benchmark.

        baseline must implement: reset(), teach(text), query(text) -> str
        """
        rng = random.Random(42)
        all_predictions = []
        all_targets = []
        all_hops = []
        all_latencies = []

        for n_hops in self.hop_values:
            for k in self.k_values:
                for chain_idx in range(self.n_chains):
                    baseline.reset()

                    # Generate chain entities
                    chain_len = n_hops + 1
                    entities = list(_ENTITIES[:max(k, chain_len + 5)])
                    while len(entities) < k:
                        entities.append(f"Entity{len(entities)}")
                    rng.shuffle(entities)

                    chain = entities[:chain_len]

                    # Teach chain facts
                    for i in range(n_hops):
                        baseline.teach(f"{chain[i]} KNOWS {chain[i+1]}")

                    # Teach distractor facts
                    distractor_ents = entities[chain_len:]
                    for de in distractor_ents[:k - n_hops]:
                        target = rng.choice(entities)
                        baseline.teach(f"{de} KNOWS {target}")

                    # Query
                    query = f"Starting from {chain[0]}, following KNOWS links {n_hops} times, who do you reach?"
                    expected = chain[-1]

                    with LatencyTracker() as lt:
                        response = baseline.query(query)

                    all_predictions.append(response)
                    all_targets.append(expected)
                    all_hops.append(n_hops)
                    all_latencies.append(lt.elapsed_ms)

        exact_match = compute_exact_match(all_predictions, all_targets)
        llm_match = compute_llm_match(all_predictions, all_targets)  # D-229
        hop_rates = compute_hop_success_rate(all_predictions, all_targets, all_hops)
        p50, p95 = compute_latency(all_latencies)

        return BenchmarkMetrics(
            exact_match=exact_match,
            llm_match=llm_match,
            hop_success_rate=hop_rates,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            total_queries=len(all_predictions),
            correct=int(llm_match * len(all_predictions)),
        )
