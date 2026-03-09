"""Atomic JSON + .pt save/load for memory persistence.

Uses temp-file-then-rename pattern to avoid corruption on crash.
Version markers ensure consistency between JSON and PT files.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .memory_bank import MemoryBank, MemoryEntry


_SNAPSHOT_VERSION_KEY = "snapshot_version"


def save_memory(bank: MemoryBank, json_path: str, pt_path: Optional[str] = None):
    """Atomically save memory bank to JSON + .pt files.

    Args:
        bank: MemoryBank instance to save
        json_path: Path for JSON metadata file
        pt_path: Path for PyTorch tensor file (defaults to json_path + '.pt')
    """
    if pt_path is None:
        pt_path = json_path + ".pt"

    keys, values, metadata = bank.get_snapshot()
    version = int(time.time() * 1000)

    # Serialize metadata to JSON-compatible dicts
    meta_dicts = []
    for entry in metadata:
        meta_dicts.append({
            "text": entry.text,
            "mem_type": entry.mem_type,
            "subject": entry.subject,
            "timestamp": entry.timestamp,
            "access_count": entry.access_count,
            "extra": entry.extra,
        })

    # Serialize graph edges
    with bank._lock:
        edges_snapshot = {k: list(v) for k, v in bank._edges.items()}

    json_data = {
        "values": [entry.text for entry in metadata],
        "metadata": meta_dicts,
        "edges": edges_snapshot,
        _SNAPSHOT_VERSION_KEY: version,
    }

    # Build tensor dict
    pt_data = {_SNAPSHOT_VERSION_KEY: version}
    if keys:
        pt_data["keys"] = torch.stack(keys)
        pt_data["values"] = torch.stack(values)
    else:
        pt_data["keys"] = torch.zeros(0, bank.d_key)
        pt_data["values"] = torch.zeros(0, bank.d_val)

    # Atomic write: temp file then rename
    json_dir = os.path.dirname(json_path) or "."
    os.makedirs(json_dir, exist_ok=True)

    # Write JSON
    fd, tmp_json = tempfile.mkstemp(dir=json_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False)
        _safe_replace(tmp_json, json_path)
    except Exception:
        _safe_remove(tmp_json)
        raise

    # Write PT
    pt_dir = os.path.dirname(pt_path) or "."
    os.makedirs(pt_dir, exist_ok=True)
    fd, tmp_pt = tempfile.mkstemp(dir=pt_dir, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(pt_data, tmp_pt)
        _safe_replace(tmp_pt, pt_path)
    except Exception:
        _safe_remove(tmp_pt)
        raise

    bank.mark_clean()


def load_memory(bank: MemoryBank, json_path: str, pt_path: Optional[str] = None) -> bool:
    """Load memory bank from JSON + .pt files.

    Returns True if loaded successfully, False if files don't exist.
    """
    if pt_path is None:
        pt_path = json_path + ".pt"

    if not os.path.exists(json_path) or not os.path.exists(pt_path):
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    pt_data = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Version consistency check
    json_version = json_data.get(_SNAPSHOT_VERSION_KEY, 0)
    pt_version = pt_data.get(_SNAPSHOT_VERSION_KEY, 0)
    if json_version != pt_version:
        return False

    # Reconstruct entries
    meta_dicts = json_data.get("metadata", [])
    keys_tensor = pt_data.get("keys", torch.zeros(0, bank.d_key))
    values_tensor = pt_data.get("values", torch.zeros(0, bank.d_val))

    if keys_tensor.shape[0] != len(meta_dicts):
        return False

    # Guard against checkpoint dimension mismatch (e.g. d_key changed in config)
    if keys_tensor.shape[0] > 0 and keys_tensor.shape[1] != bank.d_key:
        return False
    if values_tensor.shape[0] > 0 and values_tensor.shape[1] != bank.d_val:
        return False

    keys = [keys_tensor[i] for i in range(keys_tensor.shape[0])]
    values = [values_tensor[i] for i in range(values_tensor.shape[0])]
    metadata = []
    for md in meta_dicts:
        metadata.append(MemoryEntry(
            text=md.get("text", ""),
            mem_type=md.get("mem_type", "fact"),
            subject=md.get("subject", ""),
            timestamp=md.get("timestamp", 0.0),
            access_count=md.get("access_count", 0),
            extra=md.get("extra", {}),
        ))

    bank.load_snapshot(keys, values, metadata)

    # Restore graph edges if present
    edges = json_data.get("edges", {})
    if edges:
        with bank._lock:
            bank._edges = {k: list(v) for k, v in edges.items()}

    return True


def _safe_replace(src: str, dst: str):
    """Atomically replace dst with src. os.replace is atomic on all platforms."""
    os.replace(src, dst)


def _safe_remove(path: str):
    """Remove file if it exists, silently ignoring errors."""
    try:
        os.remove(path)
    except OSError:
        pass
