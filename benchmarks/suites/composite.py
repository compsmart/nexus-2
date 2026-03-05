"""Composite benchmark: mixed-mode session with recall + multihop + reasoning + learning.

Evolved from vs_rag suite with standardised category scoring.
"""

import random
from typing import Dict, List

from ..metrics import (
    BenchmarkMetrics, LatencyTracker,
    compute_exact_match, compute_hop_success_rate, compute_latency,
)

_COMPOSITE_CASES = [
    # Memory recall
    {"q": "What does Alice like?", "target": "blue", "category": "memory",
     "teach": ["Alice likes blue.", "Bob likes red.", "Charlie likes green."]},
    {"q": "Where does Bob work?", "target": "acme", "category": "memory",
     "teach": ["Bob works at Acme.", "Alice works at Globex.", "Diana works at Initech."]},
    {"q": "What pet does Charlie own?", "target": "parrot", "category": "memory",
     "teach": ["Charlie owns a parrot.", "Diana owns a cat.", "Edward owns a dog."]},
    {"q": "What city does Diana live in?", "target": "tokyo", "category": "memory",
     "teach": ["Diana lives in Tokyo.", "Edward lives in Berlin.", "Fiona lives in Paris."]},
    {"q": "What instrument does Edward play?", "target": "violin", "category": "memory",
     "teach": ["Edward plays the violin.", "Fiona plays the piano.", "George plays guitar."]},

    # Multi-hop
    {"q": "Alice knows Bob, Bob knows Charlie. Following 2 links from Alice, who do you reach?",
     "target": "charlie", "category": "multi_hop", "hops": 2,
     "teach": ["Alice KNOWS Bob.", "Bob KNOWS Charlie.", "Charlie KNOWS Diana."]},
    {"q": "X trusts Y, Y trusts Z, Z trusts W. X=Diana, Y=Edward, Z=Fiona, W=George. Who does Diana reach in 3 steps?",
     "target": "george", "category": "multi_hop", "hops": 3,
     "teach": ["Diana TRUSTS Edward.", "Edward TRUSTS Fiona.", "Fiona TRUSTS George."]},
    {"q": "Alpha links to Bravo, Bravo links to Charlie. Following 2 links from Alpha?",
     "target": "charlie", "category": "multi_hop", "hops": 2,
     "teach": ["Alpha LINKS Bravo.", "Bravo LINKS Charlie.", "Delta LINKS Echo."]},
    {"q": "P1 knows P2, P2 knows P3, P3 knows P4. Following 3 links from P1?",
     "target": "p4", "category": "multi_hop", "hops": 3,
     "teach": ["P1 KNOWS P2.", "P2 KNOWS P3.", "P3 KNOWS P4."]},
    {"q": "Sam befriends Tom, Tom befriends Uma. Following 2 links from Sam?",
     "target": "uma", "category": "multi_hop", "hops": 2,
     "teach": ["Sam BEFRIENDS Tom.", "Tom BEFRIENDS Uma.", "Uma BEFRIENDS Vera."]},

    # Reasoning
    {"q": "If all birds can fly and a penguin is a bird, can a penguin fly according to this rule?",
     "target": "yes", "category": "reasoning",
     "teach": ["Rule: All birds can fly."]},
    {"q": "X is taller than Y, Y is taller than Z. Is X taller than Z?",
     "target": "yes", "category": "reasoning",
     "teach": ["X is taller than Y.", "Y is taller than Z."]},
    {"q": "If A implies B and B implies C, does A imply C?",
     "target": "yes", "category": "reasoning",
     "teach": ["If A then B.", "If B then C."]},
    {"q": "Three boxes: red, blue, green. Key is not in red, not in green. Where is the key?",
     "target": "blue", "category": "reasoning",
     "teach": ["There are three boxes: red, blue, green.", "Key is not in red.", "Key is not in green."]},
    {"q": "A farmer has 17 sheep. All but 9 run away. How many are left?",
     "target": "9", "category": "reasoning", "teach": []},

    # Learning
    {"q": "Using Rule A, what is CODE(dog)?", "target": "EPH", "category": "learning",
     "teach": ["Rule A: CODE(x) shifts each letter forward by one and uppercases output. Example: CODE(cat)=DBU."]},
    {"q": "Which city is the owner of Project Kappa located in?", "target": "seoul", "category": "learning",
     "teach": ["Project Kappa is owned by Lina.", "Lina is located in Seoul."]},
    {"q": "What is the emergency protocol token?", "target": "channel-7", "category": "learning",
     "teach": ["Emergency protocol token is channel-7."]},
    {"q": "Using Rule A, what is CODE(hi)?", "target": "IJ", "category": "learning",
     "teach": ["Rule A: CODE(x) shifts each letter forward by one and uppercases output. Example: CODE(cat)=DBU."]},
    {"q": "Who owns Project Kappa?", "target": "lina", "category": "learning",
     "teach": ["Project Kappa is owned by Lina.", "Lina is located in Seoul."]},
]


class CompositeSuite:
    """Composite benchmark: mixed-mode session testing memory, reasoning, multi-hop, learning."""

    def __init__(self, n_facts: int = 50, n_queries: int = 20, category_weights: Dict[str, float] = None, **kwargs):
        self.n_facts = n_facts
        self.n_queries = n_queries
        self.category_weights = category_weights or {
            "reasoning": 0.30, "memory": 0.25, "multi_hop": 0.25, "learning": 0.20,
        }

    def run(self, baseline) -> BenchmarkMetrics:
        all_predictions: List[str] = []
        all_targets: List[str] = []
        all_hops: List[int] = []
        all_latencies: List[float] = []
        category_results: Dict[str, List[bool]] = {}

        baseline.reset()

        # Teach all context
        taught = set()
        for case in _COMPOSITE_CASES:
            for doc in case.get("teach", []):
                if doc not in taught:
                    baseline.teach(doc)
                    taught.add(doc)

        # Query phase
        for case in _COMPOSITE_CASES:
            target = case["target"]
            cat = case["category"]
            hops = case.get("hops", 1)

            with LatencyTracker() as lt:
                response = baseline.query(case["q"])

            all_predictions.append(response)
            all_targets.append(target)
            all_hops.append(hops)
            all_latencies.append(lt.elapsed_ms)

            hit = target.lower() in response.lower()
            category_results.setdefault(cat, []).append(hit)

        # Compute accuracy
        exact_match = compute_exact_match(all_predictions, all_targets)
        hop_rates = compute_hop_success_rate(all_predictions, all_targets, all_hops)
        p50, p95 = compute_latency(all_latencies)

        # Compute weighted composite score
        cat_acc = {}
        for cat, results in category_results.items():
            cat_acc[cat] = sum(results) / len(results) if results else 0.0

        categories = sorted(cat_acc.keys())
        total_w = sum(self.category_weights.get(c, 0.0) for c in categories) or 1.0
        composite_score = sum(
            cat_acc.get(c, 0.0) * self.category_weights.get(c, 0.0) for c in categories
        ) / total_w

        return BenchmarkMetrics(
            exact_match=composite_score,
            hop_success_rate=hop_rates,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            total_queries=len(all_predictions),
            correct=int(exact_match * len(all_predictions)),
        )
