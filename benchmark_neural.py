#!/usr/bin/env python3
"""Neural-component benchmark for the 5 lab-derived improvements.

Tests encoder quality, multi-hop reasoning, confidence gating, and scalability
at the tensor level — no LLM required.

Usage:
    python3 benchmark_neural.py
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus2.config import NexusConfig
from nexus2.memory.encoder import LSTMEncoder
from nexus2.memory.memory_bank import MemoryBank
from nexus2.reasoning.chain_executor import ChainExecutor, ReasoningResult
from nexus2.reasoning.confidence_gate import (
    ConfidenceGate,
    MultiSignalConfidenceGate,
    RouteLevel,
)
from nexus2.learning.curriculum_engine import CurriculumEngine


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_embedding(config: NexusConfig, device: str) -> nn.Embedding:
    return nn.Embedding(config.vocab_size, config.embed_dim).to(device)


def _make_encoder(config: NexusConfig, device: str) -> nn.Module:
    """Build encoder based on config.encoder_type."""
    if config.encoder_type == "mamba":
        try:
            from nexus2.memory.mamba_encoder import MambaEncoder, MAMBA_AVAILABLE
            if MAMBA_AVAILABLE and torch.cuda.is_available():
                return MambaEncoder(
                    embed_dim=config.embed_dim,
                    hidden_dim=config.lstm_hidden,
                    d_key=config.d_key,
                    d_val=config.d_val,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand,
                ).to(device)
        except ImportError:
            pass
    return LSTMEncoder(
        embed_dim=config.embed_dim,
        hidden_dim=config.lstm_hidden,
        d_key=config.d_key,
        d_val=config.d_val,
        num_layers=config.lstm_layers,
        dropout=config.encoder_dropout,
    ).to(device)


def _tokenize(text: str, vocab_size: int, device: str) -> torch.Tensor:
    tokens = [hash(c) % vocab_size for c in text]
    return torch.tensor([tokens or [0]], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# 1. Encoder Quality Benchmark
# ---------------------------------------------------------------------------

def bench_encoder(device: str) -> Dict:
    """Compare LSTM vs Mamba encoder on key/value separability at various k."""
    results = {}

    for enc_type in ["lstm", "mamba"]:
        cfg = NexusConfig()
        cfg.encoder_type = enc_type
        cfg.d_key = 512
        cfg.d_val = 512
        cfg.embed_dim = 128
        cfg.lstm_hidden = 512
        cfg.vocab_size = 2000

        try:
            enc_device = "cuda" if (enc_type == "mamba" and torch.cuda.is_available()) else device
            encoder = _make_encoder(cfg, enc_device)
            embedding = _make_embedding(cfg, enc_device)
        except Exception:
            results[enc_type] = {"status": "unavailable"}
            continue

        encoder.eval()
        embedding.eval()

        enc_results = {}
        for k in [10, 50, 100, 200, 500]:
            # Generate k unique facts
            rng = random.Random(42)
            entities = [f"Entity_{i}" for i in range(k)]
            attrs = [f"attr_{rng.randint(0, 999)}" for _ in range(k)]
            facts = [f"{e} has {a}" for e, a in zip(entities, attrs)]

            # Encode all facts
            with torch.no_grad():
                keys = []
                for fact in facts:
                    tokens = _tokenize(fact, cfg.vocab_size, enc_device)
                    embeds = embedding(tokens)
                    key, _ = encoder.encode_single(embeds)
                    keys.append(key.squeeze(0))
                key_matrix = torch.stack(keys)  # [k, d_key]

            # Metric: avg cosine similarity between all pairs (lower = better separation)
            key_norm = F.normalize(key_matrix, dim=-1)
            sim_matrix = key_norm @ key_norm.T
            # Mask diagonal
            mask = ~torch.eye(k, dtype=torch.bool, device=enc_device)
            avg_off_diag = sim_matrix[mask].mean().item()

            # Metric: how often does the correct key rank #1 for self-query?
            # (identity retrieval accuracy)
            top1_hits = 0
            for i in range(k):
                sims = sim_matrix[i].clone()
                sims[i] = -1  # mask self
                # Check how similar the nearest neighbor is vs self
                top1_hits += 1  # self always matches self; measure uniqueness instead

            enc_results[k] = {
                "avg_off_diag_sim": round(avg_off_diag, 4),
                "key_norm_mean": round(key_matrix.norm(dim=-1).mean().item(), 4),
                "key_norm_std": round(key_matrix.norm(dim=-1).std().item(), 4),
            }

        results[enc_type] = enc_results

    return results


# ---------------------------------------------------------------------------
# 2. Multi-Hop Reasoning Accuracy (untrained, measuring loss landscape)
# ---------------------------------------------------------------------------

def bench_multihop(device: str) -> Dict:
    """Measure reasoning chain loss at different d_key/d_val and hop depths."""
    results = {}

    for d in [256, 512]:
        cfg = NexusConfig()
        cfg.encoder_type = "lstm"
        cfg.d_key = d
        cfg.d_val = d
        cfg.embed_dim = 128
        cfg.lstm_hidden = 512
        cfg.vocab_size = 2000
        cfg.max_hops = 5
        cfg.entropy_lambda = 0.01 if d == 512 else 0.0

        embedding = _make_embedding(cfg, device)
        encoder = _make_encoder(cfg, device)
        chain = ChainExecutor(cfg, n_entities=cfg.vocab_size).to(device)

        d_results = {}
        rng = random.Random(42)

        for n_hops in [2, 3, 4, 5]:
            losses = []
            for trial in range(20):
                # Generate random chain
                k = 50
                entities = [f"E{i}" for i in range(k)]
                chain_ents = rng.sample(entities, n_hops + 1)

                with torch.no_grad():
                    all_keys, all_vals = [], []
                    for e in entities:
                        tokens = _tokenize(e, cfg.vocab_size, device)
                        embeds = embedding(tokens)
                        key, val = encoder.encode_single(embeds)
                        all_keys.append(key)
                        all_vals.append(val)

                    mem_keys = torch.cat(all_keys, dim=0).unsqueeze(0)
                    mem_vals = torch.cat(all_vals, dim=0).unsqueeze(0)

                    query_tokens = _tokenize(chain_ents[0], cfg.vocab_size, device)
                    query_embeds = embedding(query_tokens)
                    _, query_val = encoder.encode_single(query_embeds)

                # Forward with grad for loss
                result = chain(query_val, mem_keys, mem_vals, n_hops=n_hops)
                targets = [rng.randint(0, cfg.vocab_size - 1) for _ in range(n_hops)]
                loss = chain.compute_loss(result, targets, supervision_weight=1.0)
                losses.append(loss.item())

            d_results[f"{n_hops}_hop"] = {
                "mean_loss": round(sum(losses) / len(losses), 4),
                "min_loss": round(min(losses), 4),
                "max_loss": round(max(losses), 4),
            }

        results[f"d={d}_entropy={'on' if cfg.entropy_lambda > 0 else 'off'}"] = d_results

    return results


# ---------------------------------------------------------------------------
# 3. Confidence Gate Precision
# ---------------------------------------------------------------------------

def bench_gate_precision() -> Dict:
    """Compare single-signal vs multi-signal gate routing accuracy."""
    rng = random.Random(42)
    results = {}

    # Simulate scenarios with known ground truth
    scenarios = []
    for _ in range(200):
        max_attn = rng.uniform(0.0, 1.0)
        margin = rng.uniform(0.0, 0.5)
        age = rng.uniform(0, 500000)
        mem_type = rng.choice(["fact", "identity", "user_input"])
        # Ground truth: should inject if attn > 0.4 AND (margin > 0.1 OR identity)
        should_inject = max_attn > 0.4 and (margin > 0.1 or mem_type == "identity")
        scenarios.append((max_attn, margin, age, mem_type, should_inject))

    # Single-signal gate
    gate_old = ConfidenceGate(threshold=0.5)
    old_correct = 0
    for max_attn, margin, age, mem_type, gt in scenarios:
        route, _ = gate_old.route(torch.tensor([max_attn]))
        predicted_inject = route == "known"
        if predicted_inject == gt:
            old_correct += 1

    # Multi-signal gate
    gate_new = MultiSignalConfidenceGate(
        low_threshold=0.30, high_threshold=0.55,
        margin_threshold=0.10, stale_seconds=259200.0,
    )
    new_correct = 0
    route_dist = {level.value: 0 for level in RouteLevel}
    for max_attn, margin, age, mem_type, gt in scenarios:
        scores = torch.tensor([max_attn, max(0, max_attn - margin)])
        level, _, signals = gate_new.route(
            torch.tensor([max_attn]),
            retrieval_scores=scores,
            top_entry_age=age,
            top_entry_type=mem_type,
        )
        route_dist[level.value] += 1
        predicted_inject = level != RouteLevel.SKIP
        if predicted_inject == gt:
            new_correct += 1

    results["single_signal"] = {
        "accuracy": round(old_correct / len(scenarios), 4),
        "route_dist": {"known": sum(1 for _, _, _, _, gt in scenarios if gt),
                       "novel": sum(1 for _, _, _, _, gt in scenarios if not gt)},
    }
    results["multi_signal"] = {
        "accuracy": round(new_correct / len(scenarios), 4),
        "route_distribution": route_dist,
    }
    results["improvement"] = round(
        (new_correct - old_correct) / len(scenarios) * 100, 2
    )

    return results


# ---------------------------------------------------------------------------
# 4. k-Schedule Safety & Scalability
# ---------------------------------------------------------------------------

def bench_k_schedule() -> Dict:
    """Verify LSTM caps and Mamba extends; measure schedule properties."""
    results = {}

    for enc_type in ["lstm", "mamba"]:
        engine = CurriculumEngine(
            k_schedule=[5, 10, 20, 50, 100, 200, 350, 500, 750, 1000],
            encoder_type=enc_type,
        )
        results[enc_type] = {
            "k_schedule": engine.k_schedule,
            "max_k": max(engine.k_schedule),
            "num_stages": len(engine.k_schedule),
            "safety_cap_active": max(engine.k_schedule) <= 500,
        }

    return results


# ---------------------------------------------------------------------------
# 5. Attention Entropy Penalty Effect
# ---------------------------------------------------------------------------

def bench_entropy_penalty(device: str) -> Dict:
    """Measure how entropy penalty affects attention distributions."""
    results = {}

    for entropy_lambda in [0.0, 0.01, 0.05]:
        cfg = NexusConfig()
        cfg.encoder_type = "lstm"
        cfg.d_key = 256
        cfg.d_val = 256
        cfg.embed_dim = 128
        cfg.lstm_hidden = 512
        cfg.vocab_size = 2000
        cfg.max_hops = 3
        cfg.entropy_lambda = entropy_lambda

        embedding = _make_embedding(cfg, device)
        encoder = _make_encoder(cfg, device)
        chain = ChainExecutor(cfg, n_entities=cfg.vocab_size).to(device)
        rng = random.Random(42)

        entropies = []
        losses = []
        for trial in range(30):
            k = 20
            with torch.no_grad():
                all_keys, all_vals = [], []
                for i in range(k):
                    tokens = _tokenize(f"fact_{i}_{rng.randint(0,999)}", cfg.vocab_size, device)
                    embeds = embedding(tokens)
                    key, val = encoder.encode_single(embeds)
                    all_keys.append(key)
                    all_vals.append(val)

                mem_keys = torch.cat(all_keys, dim=0).unsqueeze(0)
                mem_vals = torch.cat(all_vals, dim=0).unsqueeze(0)

                tokens = _tokenize(f"query_{trial}", cfg.vocab_size, device)
                embeds = embedding(tokens)
                _, query_val = encoder.encode_single(embeds)

            result = chain(query_val, mem_keys, mem_vals, n_hops=3)
            targets = [rng.randint(0, cfg.vocab_size - 1) for _ in range(3)]
            loss = chain.compute_loss(result, targets, supervision_weight=1.0)
            losses.append(loss.item())

            # Measure entropy of attention weights
            for aw in result.attention_weights:
                H = -(aw * torch.log(aw + 1e-8)).sum(dim=-1).mean().item()
                entropies.append(H)

        results[f"lambda={entropy_lambda}"] = {
            "avg_entropy": round(sum(entropies) / len(entropies), 4),
            "avg_loss": round(sum(losses) / len(losses), 4),
            "min_entropy": round(min(entropies), 4),
            "max_entropy": round(max(entropies), 4),
        }

    return results


# ---------------------------------------------------------------------------
# 6. Encoding Latency: LSTM vs Mamba
# ---------------------------------------------------------------------------

def bench_encoder_latency(device: str) -> Dict:
    """Measure encoding latency for different sequence lengths."""
    results = {}

    for enc_type in ["lstm", "mamba"]:
        cfg = NexusConfig()
        cfg.encoder_type = enc_type
        cfg.d_key = 512
        cfg.d_val = 512
        cfg.embed_dim = 128
        cfg.lstm_hidden = 512
        cfg.vocab_size = 2000

        try:
            enc_device = "cuda" if (enc_type == "mamba" and torch.cuda.is_available()) else device
            encoder = _make_encoder(cfg, enc_device)
            embedding = _make_embedding(cfg, enc_device)
        except Exception:
            results[enc_type] = {"status": "unavailable"}
            continue

        encoder.eval()
        embedding.eval()

        lat_results = {}
        for seq_len in [16, 64, 128, 256, 512]:
            latencies = []
            for _ in range(50):
                tokens = torch.randint(0, cfg.vocab_size, (1, seq_len), device=enc_device)
                embeds = embedding(tokens)

                if enc_device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.no_grad():
                    encoder.encode_single(embeds)

                if enc_device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

            sorted_lat = sorted(latencies)
            lat_results[f"seq={seq_len}"] = {
                "p50_ms": round(sorted_lat[len(sorted_lat) // 2], 3),
                "p95_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 3),
            }

        results[enc_type] = lat_results

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_section(title: str, data: Dict, indent: int = 0):
    prefix = "  " * indent
    print(f"{prefix}{title}")
    for k, v in data.items():
        if isinstance(v, dict):
            _print_section(f"{k}:", v, indent + 1)
        else:
            print(f"{prefix}  {k}: {v}")


def main():
    device = "cpu"  # LSTM tests on CPU; Mamba auto-uses CUDA
    print("=" * 70)
    print("NEXUS-2 Neural Component Benchmark")
    print("  5 Lab-Derived Improvements: D-184, D-195, D-197, D-188, D-199")
    print("=" * 70)

    # Config verification
    cfg = NexusConfig()
    print(f"\nConfig: encoder_type={cfg.encoder_type}, d_key={cfg.d_key}, "
          f"d_val={cfg.d_val}, entropy_lambda={cfg.entropy_lambda}")
    print(f"k_schedule={cfg.k_schedule}")
    print()

    # 1. Encoder Quality
    print("-" * 70)
    print("1. ENCODER KEY SEPARABILITY (D-184 Mamba + D-199 d=512)")
    print("-" * 70)
    enc_results = bench_encoder(device)
    _print_section("", enc_results)
    print()

    # 2. Multi-Hop Reasoning
    print("-" * 70)
    print("2. MULTI-HOP REASONING LOSS (D-199 d=256→512 + D-195 entropy)")
    print("-" * 70)
    hop_results = bench_multihop(device)
    _print_section("", hop_results)
    print()

    # 3. Confidence Gate
    print("-" * 70)
    print("3. CONFIDENCE GATE PRECISION (D-197 Multi-Signal)")
    print("-" * 70)
    gate_results = bench_gate_precision()
    _print_section("", gate_results)
    print()

    # 4. k-Schedule
    print("-" * 70)
    print("4. K-SCHEDULE SAFETY & SCALABILITY (D-188)")
    print("-" * 70)
    k_results = bench_k_schedule()
    _print_section("", k_results)
    print()

    # 5. Entropy Penalty
    print("-" * 70)
    print("5. ATTENTION ENTROPY PENALTY EFFECT (D-195)")
    print("-" * 70)
    entropy_results = bench_entropy_penalty(device)
    _print_section("", entropy_results)
    print()

    # 6. Encoding Latency
    print("-" * 70)
    print("6. ENCODER LATENCY (D-184 LSTM vs Mamba)")
    print("-" * 70)
    lat_results = bench_encoder_latency(device)
    _print_section("", lat_results)
    print()

    print("=" * 70)
    print("Benchmark complete.")
    print("=" * 70)

    return {
        "encoder_quality": enc_results,
        "multihop": hop_results,
        "gate_precision": gate_results,
        "k_schedule": k_results,
        "entropy_penalty": entropy_results,
        "encoder_latency": lat_results,
    }


if __name__ == "__main__":
    main()
