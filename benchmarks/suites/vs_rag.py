"""Head-to-head comparison: NEXUS-2 vs RAG+Phi vs raw Phi vs NEXUS-1."""

import random
from typing import List

from ..metrics import (
    BenchmarkMetrics, LatencyTracker,
    compute_exact_match, compute_llm_match, compute_hop_success_rate, compute_latency,
)


class VsRagSuite:
    """Comprehensive comparison across memory recall and multi-hop tasks."""

    def __init__(self, n_facts: int = 50, n_queries: int = 20):
        self.n_facts = n_facts
        self.n_queries = n_queries

    def run(self, baseline) -> BenchmarkMetrics:
        rng = random.Random(42)
        all_predictions = []
        all_targets = []
        all_hops = []
        all_latencies = []

        baseline.reset()

        # Phase 1: Seed facts
        entities = [f"Person_{i}" for i in range(self.n_facts)]
        attrs = ["red", "blue", "green", "gold", "silver"]
        fact_map = {}
        for ent in entities:
            attr = rng.choice(attrs)
            fact_map[ent] = attr
            baseline.teach(f"{ent} LIKES {attr}")

        # Phase 2: 2-hop chains
        chains = []
        for i in range(0, min(20, self.n_facts - 2), 2):
            a, b, c = entities[i], entities[i + 1], entities[i + 2]
            baseline.teach(f"{a} KNOWS {b}")
            baseline.teach(f"{b} KNOWS {c}")
            chains.append((a, c, 2))

        # Phase 3: 3-hop chains
        for i in range(0, min(12, self.n_facts - 3), 3):
            a, b, c, d = entities[i], entities[i + 1], entities[i + 2], entities[i + 3]
            baseline.teach(f"{a} TRUSTS {b}")
            baseline.teach(f"{b} TRUSTS {c}")
            baseline.teach(f"{c} TRUSTS {d}")
            chains.append((a, d, 3))

        # Phase 4: Single-hop recall queries
        query_ents = rng.sample(entities, min(self.n_queries, len(entities)))
        for ent in query_ents:
            target = fact_map[ent]
            with LatencyTracker() as lt:
                response = baseline.query(f"What does {ent} like?")
            all_predictions.append(response)
            all_targets.append(target)
            all_hops.append(1)
            all_latencies.append(lt.elapsed_ms)

        # Phase 5: Multi-hop queries
        for start, end, n_hops in chains[:self.n_queries]:
            with LatencyTracker() as lt:
                response = baseline.query(
                    f"Following {n_hops} links from {start}, who do you reach?"
                )
            all_predictions.append(response)
            all_targets.append(end)
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
