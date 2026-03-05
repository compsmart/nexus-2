"""CLI entry point for training NEXUS-2 neural components.

Usage:
    python train.py --phase all        # Full pipeline
    python train.py --phase encoder    # Just encoder k-curriculum
    python train.py --phase hops       # Just multi-hop chain
    python train.py --phase adapter    # Just soft-prompt adapter
    python train.py --phase distill    # Encoder -> Conv1D distillation
"""

import argparse
import os
import sys
import time

import torch

from nexus2.config import NexusConfig
from nexus2.learning.trainer import NexusTrainer


def main():
    parser = argparse.ArgumentParser(description="NEXUS-2 Training Pipeline")
    parser.add_argument(
        "--phase",
        choices=["all", "encoder", "hops", "distill", "adapter"],
        default="all",
        help="Training phase to run",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--encoder-type", default="lstm",
                        help="Encoder type (lstm recommended for full pipeline incl. distillation)")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--k-max", type=int, default=None,
                        help="Cap k-schedule at this value (e.g. 50 for quick training)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    config = NexusConfig()
    config.encoder_type = args.encoder_type
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.k_max:
        config.k_schedule = [k for k in config.k_schedule if k <= args.k_max]

    print(f"\nConfig:", flush=True)
    print(f"  encoder_type:   {config.encoder_type}", flush=True)
    print(f"  d_key/d_val:    {config.d_key}/{config.d_val}", flush=True)
    print(f"  entropy_lambda: {config.entropy_lambda}", flush=True)
    print(f"  k_schedule:     {config.k_schedule}", flush=True)
    print(f"  checkpoint_dir: {config.checkpoint_dir}", flush=True)
    print(flush=True)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    t0 = time.time()
    trainer = NexusTrainer(config, device=device)

    if args.phase == "all":
        trainer.train_all()
    elif args.phase == "encoder":
        trainer.train_encoder()
        trainer.save_checkpoints()
    elif args.phase == "hops":
        trainer.train_hops()
        trainer.save_checkpoints()
    elif args.phase == "distill":
        trainer.train_distill()
        trainer.save_checkpoints()
    elif args.phase == "adapter":
        trainer.train_adapter()
        trainer.save_checkpoints()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}m)", flush=True)

    # Verify checkpoints
    for name in ["embedding.pt", "encoder.pt", "chain.pt", "conv_encoder.pt", "adapter.pt"]:
        path = os.path.join(config.checkpoint_dir, name)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  {name}: {size:.0f} KB", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
