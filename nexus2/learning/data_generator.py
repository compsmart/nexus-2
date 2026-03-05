"""Synthetic data generators for training.

Generates fact-recall and multi-hop chain data with diverse entity names.

ANTI-PATTERN: NEVER use shared-prefix entity names (use diverse adjective+noun).
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Diverse adjective+noun vocabulary for entity names (no shared prefixes)
_ADJECTIVES = [
    "crimson", "golden", "silver", "azure", "jade", "coral", "amber",
    "violet", "scarlet", "ivory", "cobalt", "bronze", "emerald", "onyx",
    "pearl", "rustic", "frozen", "molten", "silent", "cosmic", "dusty",
    "nimble", "gentle", "fierce", "hollow", "bright", "ancient", "bold",
    "calm", "dapper", "eager", "frosty", "grand", "hazy", "jolly",
    "keen", "lofty", "merry", "noble", "proud", "quiet", "rapid",
    "swift", "tidy", "vivid", "warm", "young", "zesty", "lunar",
]

_NOUNS = [
    "falcon", "panther", "river", "summit", "beacon", "crystal", "harbor",
    "meadow", "thunder", "whisper", "canyon", "glacier", "phoenix", "tempest",
    "lantern", "compass", "anchor", "sparrow", "cedar", "marble", "atlas",
    "bison", "comet", "delta", "ember", "flint", "grove", "heron",
    "iris", "jasper", "kite", "lotus", "maple", "nexus", "orbit",
    "prism", "quartz", "reef", "sage", "tulip", "umber", "vortex",
    "wren", "zenith", "arrow", "brook", "crest", "dusk", "elm",
]

_RELATIONS = [
    "LIKES", "KNOWS", "TRUSTS", "FOLLOWS", "TEACHES", "HELPS",
    "ADMIRES", "VISITS", "CALLS", "JOINS",
]

_ATTRIBUTES = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink",
    "black", "white", "gray", "brown", "teal", "navy", "lime",
    "gold", "cyan", "plum", "sage", "rust", "cream",
]


def _generate_entity_names(n: int, rng: random.Random) -> List[str]:
    """Generate n diverse entity names (adjective + noun, no shared prefixes)."""
    adj_pool = list(_ADJECTIVES)
    noun_pool = list(_NOUNS)
    rng.shuffle(adj_pool)
    rng.shuffle(noun_pool)

    names = set()
    while len(names) < n:
        adj = adj_pool[len(names) % len(adj_pool)]
        noun = noun_pool[len(names) % len(noun_pool)]
        name = f"{adj}_{noun}"
        if name not in names:
            names.add(name)
        else:
            # Fallback: random combo
            names.add(f"{rng.choice(_ADJECTIVES)}_{rng.choice(_NOUNS)}_{len(names)}")
    return list(names)


@dataclass
class FactRecallSample:
    """A single fact-recall training sample."""
    facts: List[str]                    # all facts (context)
    query_entity: str                   # entity to query
    target_attribute: str               # correct answer
    entity_to_idx: Dict[str, int]       # entity name -> vocab index
    target_idx: int                     # target entity/attribute vocab index
    tokens: List[List[int]]             # tokenized facts


@dataclass
class MultiHopSample:
    """A multi-hop chain training sample."""
    facts: List[str]                    # all facts forming the chain + distractors
    chain: List[str]                    # entity chain: [A, B, C, ..., N]
    query_entity: str                   # start entity
    intermediate_targets: List[int]     # target indices per hop
    final_target: str                   # final answer entity
    n_hops: int


class FactRecallGenerator:
    """Generates 'Entity LIKES Attribute' style fact-recall data."""

    def __init__(self, vocab_size: int = 2000, seed: int = 42):
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)

    def generate(
        self,
        k: int = 10,
        n_queries: int = 50,
    ) -> List[FactRecallSample]:
        """Generate fact-recall samples.

        Args:
            k: number of facts per sample
            n_queries: number of query samples to generate

        Returns:
            List of FactRecallSample instances.
        """
        entities = _generate_entity_names(k, self.rng)
        attributes = list(_ATTRIBUTES)

        # Assign each entity a random attribute
        entity_attr = {}
        for ent in entities:
            entity_attr[ent] = self.rng.choice(attributes)

        # Build entity -> vocab index mapping
        entity_to_idx = {}
        for i, name in enumerate(entities):
            entity_to_idx[name] = i % self.vocab_size
        for i, attr in enumerate(attributes):
            entity_to_idx[attr] = (len(entities) + i) % self.vocab_size

        # Generate facts
        relation = self.rng.choice(_RELATIONS)
        facts = [f"{ent} {relation} {entity_attr[ent]}" for ent in entities]

        # Tokenize: simple char-hash based
        def tokenize(text):
            return [hash(c) % self.vocab_size for c in text]

        fact_tokens = [tokenize(f) for f in facts]

        # Generate query samples
        samples = []
        for _ in range(n_queries):
            query_ent = self.rng.choice(entities)
            target_attr = entity_attr[query_ent]
            samples.append(FactRecallSample(
                facts=facts,
                query_entity=query_ent,
                target_attribute=target_attr,
                entity_to_idx=entity_to_idx,
                target_idx=entity_to_idx[target_attr],
                tokens=fact_tokens,
            ))

        return samples


class MultiHopChainGenerator:
    """Generates N-hop chain data: A->B->C->...->N.

    Each hop is a fact "X RELATION Y". The chain follows through intermediate
    entities: query(A) -> B -> C -> ... -> N.
    """

    def __init__(self, vocab_size: int = 2000, seed: int = 42):
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)

    def generate(
        self,
        n_hops: int = 3,
        k: int = 10,
        n_distractors: int = 5,
        n_samples: int = 50,
    ) -> List[MultiHopSample]:
        """Generate multi-hop chain samples.

        Args:
            n_hops: number of hops in the chain
            k: total entities involved (chain + distractors)
            n_distractors: extra distractor facts
            n_samples: number of samples to generate

        Returns:
            List of MultiHopSample instances.
        """
        samples = []

        for _ in range(n_samples):
            # Need n_hops + 1 entities for the chain
            chain_len = n_hops + 1
            total_entities = max(k, chain_len + n_distractors)
            entities = _generate_entity_names(total_entities, self.rng)

            # Build entity -> index mapping
            entity_to_idx = {ent: i % self.vocab_size for i, ent in enumerate(entities)}

            # Chain entities
            chain = entities[:chain_len]
            relation = self.rng.choice(_RELATIONS)

            # Chain facts
            chain_facts = [
                f"{chain[i]} {relation} {chain[i+1]}"
                for i in range(n_hops)
            ]

            # Distractor facts
            distractor_entities = entities[chain_len:]
            distractor_facts = []
            for de in distractor_entities[:n_distractors]:
                target = self.rng.choice(entities)
                distractor_facts.append(f"{de} {relation} {target}")

            # Combine and shuffle
            all_facts = chain_facts + distractor_facts
            self.rng.shuffle(all_facts)

            # Intermediate targets: entity index at each hop
            intermediate_targets = [entity_to_idx[chain[i+1]] for i in range(n_hops)]

            samples.append(MultiHopSample(
                facts=all_facts,
                chain=chain,
                query_entity=chain[0],
                intermediate_targets=intermediate_targets,
                final_target=chain[-1],
                n_hops=n_hops,
            ))

        return samples
