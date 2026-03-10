"""Cosine memory bank with cross-sequence persistence.

Core storage: parallel tensors for keys/values plus metadata dicts.
Retrieval via cosine similarity. Novelty-triggered slot growth.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# D-295: Optional FAISS for sub-linear retrieval at scale
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


@dataclass
class MemoryEntry:
    """Metadata for a single memory slot."""
    text: str = ""
    mem_type: str = "fact"
    subject: str = ""
    timestamp: float = 0.0
    access_count: int = 0
    extra: Dict = field(default_factory=dict)


class MemoryBank:
    """Append-only memory bank with cosine retrieval and temporal decay.

    Thread-safe via snapshot-before-compute pattern: copy data under lock,
    release lock, then do computation outside the lock.
    """

    def __init__(
        self,
        d_key: int = 256,
        d_val: int = 256,
        max_slots: int = 10000,
        novelty_threshold: float = 0.5,
        decay_enabled: bool = True,
        decay_half_lives: Optional[Dict[str, Optional[float]]] = None,
        dedup_enabled: bool = True,
        dedup_scope: str = "exact_text",
        type_boosts: Optional[Dict[str, float]] = None,
    ):
        self.d_key = d_key
        self.d_val = d_val
        self.max_slots = max_slots
        self.novelty_threshold = novelty_threshold
        self.decay_enabled = decay_enabled
        self.type_boosts = type_boosts or {}
        self.decay_half_lives = decay_half_lives or {
            "identity": None,
            "skill": 2_592_000.0,
            "fact": 1_209_600.0,
            "user_input": 259_200.0,
            "agent_response": 259_200.0,
            "default": 604_800.0,
        }
        self.dedup_enabled = dedup_enabled
        self.dedup_scope = dedup_scope

        self._lock = threading.Lock()
        self._keys: List[torch.Tensor] = []      # each [d_key]
        self._values: List[torch.Tensor] = []     # each [d_val]
        self._metadata: List[MemoryEntry] = []
        self._dedup_counts: Dict[Tuple[str, str], int] = {}
        self._dirty = False

        # D-295: FAISS index for sub-linear retrieval
        self._faiss_index: Optional[object] = None
        self._faiss_threshold = 50  # use FAISS when bank > this many entries
        if _FAISS_AVAILABLE:
            self._faiss_index = faiss.IndexFlatIP(d_key)

        # Graph overlay: adjacency list for entity relationships
        # Maps entity -> list of {target, relation} dicts
        self._edges: Dict[str, List[Dict[str, str]]] = {}

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._keys)

    @property
    def dirty(self) -> bool:
        return self._dirty

    def _dedup_key(self, entry: MemoryEntry) -> Tuple[str, str]:
        """Generate dedup key from entry."""
        text = entry.text
        if self.dedup_scope == "normalized_text":
            text = text.strip().lower()
        return (entry.mem_type, text)

    def _has_duplicate(self, entry: MemoryEntry) -> bool:
        if not self.dedup_enabled or self.dedup_scope == "off":
            return False
        return self._dedup_counts.get(self._dedup_key(entry), 0) > 0

    def _dedup_inc(self, entry: MemoryEntry):
        if self.dedup_enabled and self.dedup_scope != "off":
            k = self._dedup_key(entry)
            self._dedup_counts[k] = self._dedup_counts.get(k, 0) + 1

    def _dedup_dec(self, entry: MemoryEntry):
        if self.dedup_enabled and self.dedup_scope != "off":
            k = self._dedup_key(entry)
            count = self._dedup_counts.get(k, 0) - 1
            if count <= 0:
                self._dedup_counts.pop(k, None)
            else:
                self._dedup_counts[k] = count

    def _faiss_add(self, key: torch.Tensor):
        """D-295: Add a single key to the FAISS index."""
        if self._faiss_index is None:
            return
        vec = F.normalize(key.unsqueeze(0), dim=-1).cpu().numpy().astype(np.float32)
        self._faiss_index.add(vec)

    def _faiss_rebuild(self):
        """D-295: Rebuild FAISS index from all current keys."""
        if self._faiss_index is None:
            return
        self._faiss_index.reset()
        if self._keys:
            keys_t = torch.stack(self._keys)
            keys_norm = F.normalize(keys_t, dim=-1).cpu().numpy().astype(np.float32)
            self._faiss_index.add(keys_norm)

    def _decay_multiplier(self, entry: MemoryEntry) -> float:
        """Compute exponential decay multiplier for an entry."""
        if not self.decay_enabled:
            return 1.0
        half_life = self.decay_half_lives.get(
            entry.mem_type,
            self.decay_half_lives.get("default", 604_800.0),
        )
        if half_life is None:
            return 1.0
        age = time.time() - entry.timestamp
        if age <= 0:
            return 1.0
        return 0.5 ** (age / half_life)

    def _adaptive_decay_multiplier(
        self,
        entry: MemoryEntry,
        entropy: float,
        weight: float = 2.0,
    ) -> float:
        """D-263: Adaptive decay that accelerates with retrieval entropy.

        When attention entropy is high (uncertain retrieval), recent facts
        should dominate more aggressively. When low (clear match), standard
        decay applies.

        Formula: 0.5 ^ ((age / half_life) * (1 + weight * entropy))

        Identity-type memories are immune to entropy-based acceleration.
        """
        if not self.decay_enabled:
            return 1.0
        half_life = self.decay_half_lives.get(
            entry.mem_type,
            self.decay_half_lives.get("default", 604_800.0),
        )
        if half_life is None:
            return 1.0
        age = time.time() - entry.timestamp
        if age <= 0:
            return 1.0
        # Identity memories are immune to entropy scaling
        if entry.mem_type == "identity":
            return 0.5 ** (age / half_life)
        return 0.5 ** ((age / half_life) * (1.0 + weight * entropy))

    def _find_existing_slot(
        self,
        key: torch.Tensor,
        subject: str,
        mem_type: str,
    ) -> Optional[int]:
        """D-259: Find existing slot with matching subject+mem_type and high key similarity.

        Returns index of matching slot, or None if no match found.
        Must be called under self._lock.
        """
        if not subject or not self._keys:
            return None

        # Find candidates with same subject+mem_type
        candidates = []
        for i, meta in enumerate(self._metadata):
            if meta.subject == subject and meta.mem_type == mem_type:
                candidates.append(i)

        if not candidates:
            return None

        # Check cosine similarity of key against candidates
        key_norm = F.normalize(key.unsqueeze(0), dim=-1)
        for idx in candidates:
            stored_key_norm = F.normalize(self._keys[idx].unsqueeze(0), dim=-1)
            cos_sim = (key_norm @ stored_key_norm.T).item()
            if cos_sim > 0.95:
                return idx

        return None

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        text: str = "",
        mem_type: str = "fact",
        subject: str = "",
        extra: Optional[Dict] = None,
    ) -> bool:
        """Write a single fact to memory.

        D-259: Before appending, checks if an entry with matching
        subject+mem_type already exists (cosine sim > 0.95 on keys).
        If so, updates the existing slot instead of appending.

        Returns True if written/updated, False if rejected (duplicate).
        """
        entry = MemoryEntry(
            text=text,
            mem_type=mem_type,
            subject=subject,
            timestamp=time.time(),
            access_count=0,
            extra=extra or {},
        )

        key = key.detach().cpu()
        value = value.detach().cpu()

        with self._lock:
            if self._has_duplicate(entry):
                return False

            # D-259: Check for existing slot to update (dedup by subject+type+key similarity)
            existing_idx = self._find_existing_slot(key, subject, mem_type)
            if existing_idx is not None:
                # Update existing slot in-place
                old_entry = self._metadata[existing_idx]
                self._dedup_dec(old_entry)
                self._keys[existing_idx] = key
                self._values[existing_idx] = value
                self._metadata[existing_idx] = entry
                self._dedup_inc(entry)
                self._dirty = True
                self._faiss_rebuild()  # D-295: rebuild after in-place update
                return True

            # Try consolidation before FIFO eviction
            if len(self._keys) >= self.max_slots:
                # Release lock for consolidation (it acquires its own)
                pass
            else:
                self._keys.append(key)
                self._values.append(value)
                self._metadata.append(entry)
                self._dedup_inc(entry)
                self._dirty = True
                self._faiss_add(key)  # D-295
                return True

        # At capacity — consolidate first, then retry
        self.consolidate(similarity_threshold=0.90)

        with self._lock:
            # If consolidation freed slots, write normally
            if len(self._keys) < self.max_slots:
                self._keys.append(key)
                self._values.append(value)
                self._metadata.append(entry)
                self._dedup_inc(entry)
                self._dirty = True
                self._faiss_rebuild()  # D-295: rebuild after consolidation
                return True

            # Still full — FIFO eviction as last resort
            evicted = self._metadata[0]
            self._dedup_dec(evicted)
            self._keys.pop(0)
            self._values.pop(0)
            self._metadata.pop(0)

            self._keys.append(key)
            self._values.append(value)
            self._metadata.append(entry)
            self._dedup_inc(entry)
            self._dirty = True
            self._faiss_rebuild()  # D-295: rebuild after eviction

        return True

    def write_batch(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        texts: List[str],
        mem_types: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
    ) -> int:
        """Bulk write. Returns count of entries actually written."""
        n = keys.shape[0]
        mem_types = mem_types or ["fact"] * n
        subjects = subjects or [""] * n
        written = 0
        for i in range(n):
            ok = self.write(
                keys[i], values[i],
                text=texts[i] if i < len(texts) else "",
                mem_type=mem_types[i],
                subject=subjects[i],
            )
            if ok:
                written += 1
        return written

    def read(
        self,
        query_key: torch.Tensor,
        top_k: int = 10,
        entropy: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Retrieve top-k values by cosine similarity.

        Args:
            query_key: [d_key] or [1, d_key] query vector
            top_k: number of results to return
            entropy: D-263 retrieval entropy for adaptive decay (None = standard decay)

        Returns:
            values:   [top_k, d_val] retrieved value vectors
            weights:  [top_k] attention weights (cosine similarities)
            indices:  list of int indices into memory bank
        """
        query_key = query_key.detach().cpu()
        if query_key.dim() == 2:
            query_key = query_key.squeeze(0)

        # Snapshot under lock
        with self._lock:
            if len(self._keys) == 0:
                empty_v = torch.zeros(0, self.d_val)
                empty_w = torch.zeros(0)
                return empty_v, empty_w, []

            keys_snapshot = torch.stack(self._keys)     # [N, d_key]
            values_snapshot = torch.stack(self._values)  # [N, d_val]
            meta_snapshot = list(self._metadata)

        # D-295: FAISS-accelerated candidate retrieval for large banks
        n_entries = len(keys_snapshot)
        use_faiss = (
            self._faiss_index is not None
            and n_entries > self._faiss_threshold
            and self._faiss_index.ntotal == n_entries
        )

        if use_faiss:
            # Get 3x top_k candidates via FAISS, then rerank with decay/boost
            faiss_k = min(top_k * 3, n_entries)
            query_vec = F.normalize(query_key.unsqueeze(0), dim=-1).cpu().numpy().astype(np.float32)
            faiss_scores, faiss_indices = self._faiss_index.search(query_vec, faiss_k)
            candidate_indices = faiss_indices[0]
            cos_sim = torch.tensor(faiss_scores[0], dtype=torch.float32)
            candidate_meta = [meta_snapshot[i] for i in candidate_indices]

            # Apply decay and boosts to candidates only
            if entropy is not None:
                decay_factors = torch.tensor(
                    [self._adaptive_decay_multiplier(m, entropy) for m in candidate_meta],
                    dtype=cos_sim.dtype,
                )
            else:
                decay_factors = torch.tensor(
                    [self._decay_multiplier(m) for m in candidate_meta],
                    dtype=cos_sim.dtype,
                )
            if self.type_boosts:
                boost_factors = torch.tensor(
                    [self.type_boosts.get(m.mem_type, 1.0) for m in candidate_meta],
                    dtype=cos_sim.dtype,
                )
                decayed_scores = cos_sim * decay_factors * boost_factors
            else:
                decayed_scores = cos_sim * decay_factors

            actual_k = min(top_k, len(candidate_indices))
            local_scores, local_top = torch.topk(decayed_scores, actual_k)
            # Map back to global indices
            global_indices = [int(candidate_indices[i]) for i in local_top.tolist()]
            top_values = values_snapshot[global_indices]

            with self._lock:
                for idx in global_indices:
                    if idx < len(self._metadata):
                        self._metadata[idx].access_count += 1

            return top_values, local_scores, global_indices

        # Brute-force path (small bank or FAISS unavailable)
        query_norm = F.normalize(query_key.unsqueeze(0), dim=-1)
        keys_norm = F.normalize(keys_snapshot, dim=-1)
        cos_sim = (query_norm @ keys_norm.T).squeeze(0)  # [N]

        # Apply decay and type-based priority boost
        # D-263: Use adaptive decay when entropy is provided
        if entropy is not None:
            decay_factors = torch.tensor(
                [self._adaptive_decay_multiplier(m, entropy) for m in meta_snapshot],
                dtype=cos_sim.dtype,
            )
        else:
            decay_factors = torch.tensor(
                [self._decay_multiplier(m) for m in meta_snapshot],
                dtype=cos_sim.dtype,
            )
        if self.type_boosts:
            boost_factors = torch.tensor(
                [self.type_boosts.get(m.mem_type, 1.0) for m in meta_snapshot],
                dtype=cos_sim.dtype,
            )
            decayed_scores = cos_sim * decay_factors * boost_factors
        else:
            decayed_scores = cos_sim * decay_factors

        # Top-k
        actual_k = min(top_k, len(keys_snapshot))
        top_scores, top_indices = torch.topk(decayed_scores, actual_k)
        top_values = values_snapshot[top_indices]

        # Update access counts
        with self._lock:
            for idx in top_indices.tolist():
                if idx < len(self._metadata):
                    self._metadata[idx].access_count += 1

        return top_values, top_scores, top_indices.tolist()

    def read_with_metadata(
        self,
        query_key: torch.Tensor,
        top_k: int = 10,
        entropy: Optional[float] = None,
    ) -> List[Tuple[torch.Tensor, float, MemoryEntry]]:
        """Read with full metadata. Returns list of (value, score, metadata)."""
        values, scores, indices = self.read(query_key, top_k, entropy=entropy)

        results = []
        with self._lock:
            for i, idx in enumerate(indices):
                if idx < len(self._metadata):
                    results.append((
                        values[i],
                        scores[i].item(),
                        self._metadata[idx],
                    ))
        return results

    def max_similarity(self, query_key: torch.Tensor) -> float:
        """Return max cosine similarity to any stored key (for novelty detection)."""
        query_key = query_key.detach().cpu()
        if query_key.dim() == 2:
            query_key = query_key.squeeze(0)

        with self._lock:
            if len(self._keys) == 0:
                return 0.0
            keys_snapshot = torch.stack(self._keys)

        query_norm = F.normalize(query_key.unsqueeze(0), dim=-1)
        keys_norm = F.normalize(keys_snapshot, dim=-1)
        cos_sim = (query_norm @ keys_norm.T).squeeze(0)
        return cos_sim.max().item()

    def should_grow(self, query_key: torch.Tensor) -> bool:
        """Check if novelty threshold triggers slot growth."""
        return self.max_similarity(query_key) < self.novelty_threshold

    def delete_matching(self, text_pattern: str, only_types=None) -> int:
        """Delete entries whose text contains pattern. Returns count deleted.

        Args:
            text_pattern: substring to match (case-insensitive).
            only_types: optional set/list of mem_type values to restrict
                        deletion to (e.g. {"fact", "document"}).
        """
        pattern_lower = text_pattern.lower()
        deleted = 0
        with self._lock:
            surviving_keys = []
            surviving_values = []
            surviving_metadata = []
            for i, entry in enumerate(self._metadata):
                should_delete = (
                    pattern_lower in entry.text.lower()
                    and (not only_types or entry.mem_type in only_types)
                )
                if should_delete:
                    self._dedup_dec(entry)
                    deleted += 1
                else:
                    surviving_keys.append(self._keys[i])
                    surviving_values.append(self._values[i])
                    surviving_metadata.append(entry)
            if deleted > 0:
                self._keys = surviving_keys
                self._values = surviving_values
                self._metadata = surviving_metadata
                self._dirty = True
                self._faiss_rebuild()  # D-295
        return deleted

    def get_snapshot(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[MemoryEntry]]:
        """Thread-safe snapshot of all data for persistence."""
        with self._lock:
            return (
                list(self._keys),
                list(self._values),
                list(self._metadata),
            )

    def load_snapshot(
        self,
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
        metadata: List[MemoryEntry],
    ):
        """Load a snapshot into the bank (used for persistence restore)."""
        with self._lock:
            self._keys = list(keys)
            self._values = list(values)
            self._metadata = list(metadata)
            self._dedup_counts.clear()
            for entry in self._metadata:
                self._dedup_inc(entry)
            self._dirty = False
            self._faiss_rebuild()  # D-295

    def clear(self):
        """Clear all entries and edges."""
        with self._lock:
            self._keys.clear()
            self._values.clear()
            self._metadata.clear()
            self._dedup_counts.clear()
            self._edges.clear()
            self._dirty = True
            if self._faiss_index is not None:
                self._faiss_index.reset()  # D-295

    def text_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float, MemoryEntry]]:
        """Text-based retrieval via keyword matching (D-228 hybrid path).

        D-228: TME+RAG hybrid breaks 65% constraint ceiling to 85% (+20pp).
        This text search complements neural cosine retrieval for constraint-type
        queries where keyword overlap is more informative than embedding distance.

        Uses simple TF-based scoring: counts query token overlap with each
        memory entry's text, weighted by inverse frequency.

        Args:
            query: search query string
            top_k: max results to return

        Returns:
            List of (text, score, entry) tuples sorted by relevance.
        """
        query_lower = query.strip().lower()
        query_tokens = set(query_lower.split())

        if not query_tokens:
            return []

        with self._lock:
            meta_snapshot = list(self._metadata)

        if not meta_snapshot:
            return []

        # Score each entry by token overlap
        scored = []
        for entry in meta_snapshot:
            entry_lower = entry.text.lower()
            entry_tokens = set(entry_lower.split())

            # Jaccard-like overlap score
            overlap = query_tokens & entry_tokens
            if not overlap:
                continue

            score = len(overlap) / (len(query_tokens | entry_tokens) + 1e-8)

            # Boost exact substring matches
            if query_lower in entry_lower or any(
                qt in entry_lower for qt in query_tokens if len(qt) > 3
            ):
                score += 0.3

            # Apply decay
            score *= self._decay_multiplier(entry)
            scored.append((entry.text, score, entry))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def find_by_prefix(self, prefix: str) -> Optional[str]:
        """Find first memory entry whose text starts with prefix (case-insensitive).

        Used by chain traversal to locate "X KNOWS Y" facts without relying on
        ranked retrieval, which can fail when X also appears as a target in other
        distractor facts with equal text-match scores.

        Args:
            prefix: the prefix string to match (e.g. "Alpha KNOWS")

        Returns:
            The matching entry text, or None if not found.
        """
        prefix_upper = prefix.strip().upper()
        with self._lock:
            for entry in self._metadata:
                if entry.text.strip().upper().startswith(prefix_upper):
                    return entry.text.strip()
        return None

    def consolidate(self, similarity_threshold: float = 0.90) -> int:
        """Cluster similar memories and merge them, freeing slots.

        Groups memories by type, computes pairwise cosine similarity within
        each group, and merges clusters above the threshold. Merged entries
        get a centroid key, averaged value, and summary text.

        Identity-type and correction-type memories are never consolidated.

        All operations are performed under a single lock acquisition to
        prevent TOCTOU races with concurrent writes.

        Args:
            similarity_threshold: minimum cosine similarity to merge (0.0-1.0)

        Returns:
            Number of entries removed by merging.
        """
        with self._lock:
            if len(self._keys) < 2:
                return 0

            keys = list(self._keys)
            values = list(self._values)
            metadata = list(self._metadata)

            # Group indices by mem_type (skip identity and correction)
            type_groups: dict[str, list[int]] = {}
            for i, m in enumerate(metadata):
                if m.mem_type in ("identity", "correction"):
                    continue
                type_groups.setdefault(m.mem_type, []).append(i)

            # Find clusters within each type group
            to_merge: list[list[int]] = []

            for mem_type, indices in type_groups.items():
                if len(indices) < 2:
                    continue

                group_keys = torch.stack([keys[i] for i in indices])
                normed = F.normalize(group_keys, dim=-1)
                sim_matrix = normed @ normed.T

                visited = set()
                for a_pos in range(len(indices)):
                    if a_pos in visited:
                        continue
                    cluster = [indices[a_pos]]
                    visited.add(a_pos)
                    for b_pos in range(a_pos + 1, len(indices)):
                        if b_pos in visited:
                            continue
                        if sim_matrix[a_pos, b_pos].item() >= similarity_threshold:
                            cluster.append(indices[b_pos])
                            visited.add(b_pos)
                    if len(cluster) >= 2:
                        to_merge.append(cluster)

            if not to_merge:
                return 0

            # Build merged entries and collect indices to remove
            remove_indices: set[int] = set()
            new_keys: list[torch.Tensor] = []
            new_values: list[torch.Tensor] = []
            new_metadata: list[MemoryEntry] = []

            for cluster in to_merge:
                cluster_keys = torch.stack([keys[i] for i in cluster])
                cluster_values = torch.stack([values[i] for i in cluster])
                cluster_meta = [metadata[i] for i in cluster]

                centroid_key = cluster_keys.mean(dim=0)
                centroid_key = centroid_key / (centroid_key.norm() + 1e-8)
                avg_value = cluster_values.mean(dim=0)

                sorted_by_time = sorted(cluster_meta, key=lambda m: m.timestamp, reverse=True)
                summary_text = sorted_by_time[0].text
                source_texts = [m.text for m in sorted_by_time]

                merged_entry = MemoryEntry(
                    text=summary_text,
                    mem_type=sorted_by_time[0].mem_type,
                    subject=sorted_by_time[0].subject,
                    timestamp=sorted_by_time[0].timestamp,
                    access_count=sum(m.access_count for m in cluster_meta),
                    extra={
                        "consolidated": True,
                        "merged_count": len(cluster),
                        "source_texts": source_texts,
                    },
                )

                new_keys.append(centroid_key)
                new_values.append(avg_value)
                new_metadata.append(merged_entry)
                remove_indices.update(cluster)

            # Rebuild bank: keep non-merged entries + add merged entries
            total_removed = len(remove_indices) - len(to_merge)

            surviving_keys = []
            surviving_values = []
            surviving_metadata = []
            for i in range(len(self._keys)):
                if i not in remove_indices:
                    surviving_keys.append(self._keys[i])
                    surviving_values.append(self._values[i])
                    surviving_metadata.append(self._metadata[i])
                else:
                    self._dedup_dec(self._metadata[i])

            for k, v, m in zip(new_keys, new_values, new_metadata):
                surviving_keys.append(k)
                surviving_values.append(v)
                surviving_metadata.append(m)
                self._dedup_inc(m)

            self._keys = surviving_keys
            self._values = surviving_values
            self._metadata = surviving_metadata
            self._dirty = True
            self._faiss_rebuild()  # D-295

        return total_removed

    # ------------------------------------------------------------------
    # Graph overlay: edge-based multi-hop traversal
    # ------------------------------------------------------------------

    def add_edge(self, source: str, target: str, relation: str = "related") -> None:
        """Add a directed edge between two entities.

        Edges are stored bidirectionally so traversal works in both directions.

        Args:
            source: source entity name
            target: target entity name
            relation: relationship type (e.g., "knows", "likes")
        """
        with self._lock:
            self._edges.setdefault(source, []).append({
                "target": target,
                "relation": relation,
            })
            # Reverse edge for bidirectional queries
            self._edges.setdefault(target, []).append({
                "target": source,
                "relation": relation,
            })
            self._dirty = True

    def get_edges(self, entity: str) -> List[Dict[str, str]]:
        """Get all edges (outgoing + incoming) for an entity.

        Args:
            entity: entity name to query

        Returns:
            List of {"target": str, "relation": str} dicts.
        """
        with self._lock:
            return list(self._edges.get(entity, []))

    def traverse(
        self,
        start: str,
        relation: str,
        hops: int,
    ) -> List[str]:
        """Traverse the graph following edges of a specific relation.

        Performs breadth-first traversal up to `hops` steps, following
        only edges matching the specified relation. Handles cycles by
        tracking visited entities.

        Args:
            start: starting entity name
            relation: only follow edges with this relation
            hops: number of hops to traverse

        Returns:
            List of entities reachable at exactly `hops` distance.
            Empty list if no path exists.
        """
        with self._lock:
            edges_snapshot = {k: list(v) for k, v in self._edges.items()}

        # BFS with hop tracking
        current_level = {start}
        visited = {start}

        for _ in range(hops):
            next_level = set()
            for entity in current_level:
                for edge in edges_snapshot.get(entity, []):
                    if edge["relation"] == relation and edge["target"] not in visited:
                        next_level.add(edge["target"])
                        visited.add(edge["target"])
            if not next_level:
                return []
            current_level = next_level

        return sorted(current_level)

    def mark_clean(self):
        """Mark memory as persisted (not dirty)."""
        self._dirty = False
