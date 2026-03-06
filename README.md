# NEXUS-2 Agent

NEXUS-2 is the second-generation neural memory agent, replacing the V1 frozen-encoder approach with a fully trainable memory pipeline. Where V1 uses a frozen SentenceTransformer (MiniLM-L6-v2, 384-dim) and text-formatted context injection, NEXUS-2 trains its own encoder (LSTM, Mamba SSM, or distilled Conv1D, 512-dim), runs an N-hop attention-based reasoning chain over memory, and injects the result into the LLM via learned soft-prompt tokens rather than raw text. A 4-level confidence gate routes each query through the appropriate generation path based on attention strength, retrieval margin, memory age, and source type.

## Key Differences from V1

| Aspect | V1 (Nexus) | V2 (NEXUS-2) |
|--------|-----------|--------------|
| Encoder | Frozen MiniLM-L6-v2 (384d) | Trainable LSTM/Mamba/Conv1D (512d) |
| Retrieval | 2-hop text search (LLM-generated follow-up) | N-hop attention chain (up to 5 hops) |
| Confidence | Binary cosine threshold | 4-level multi-signal gate (D-197) |
| LLM injection | Text-formatted context in prompt | Soft-prompt tokens (learned adapter) |
| Training | None (inference only) | 5-phase curriculum on synthetic data |

## Architecture

The pipeline processes each user message through six stages:

1. **Perception** -- `InputProcessor` extracts entities, intent, personal facts, and correction signals
2. **Memory Write** -- New facts encoded via the trainable encoder and stored in the slot-based memory bank (FIFO eviction, temporal decay, deduplication)
3. **Reasoning** -- `ExplicitNReadChain` runs N-hop cosine-softmax attention over memory with per-hop query projections and intermediate entity supervision
4. **Confidence Gating** -- `MultiSignalConfidenceGate` evaluates four signals (max attention weight, retrieval margin, top entry age, source type) and routes to one of: INJECT_FULL, INJECT_TOP1, INJECT_HEDGED, or SKIP
5. **Generation** -- `SoftPromptAdapter` maps the reasoning output (d_val=512) into 4 soft tokens in LLM embedding space (3584-dim), prepended to the prompt for the frozen Qwen2.5-7B-Instruct backbone
6. **Learning** -- SKIP-routed queries trigger autonomous skill acquisition via web search; all interactions are stored back into memory

## Training Pipeline

NEXUS-2 components are trained on synthetic entity-relation datasets using a multi-phase curriculum:

1. **Encoder k-schedule** -- Train the encoder at increasing fact counts (k=5, 10, 20, 50, 100, 200, 350, 500, 750, 1000), advancing at 99% accuracy
2. **Hop-depth curriculum** -- Train the reasoning chain at increasing depth (2, 3, 4, 5 hops)
3. **Distillation** (optional) -- Distill the LSTM encoder into a Conv1D replacement for 28x faster inference
4. **Adapter training** -- Train the soft-prompt adapter to bridge reasoning output into LLM embedding space
5. **End-to-end validation** -- Verify the full pipeline on held-out data

## Usage

```bash
python main.py                   # Start chat loop
python main.py --no-llm          # Memory-only mode (no LLM)
python main.py --device cpu      # Force CPU inference
python train.py --phase all      # Run full training pipeline
python run_benchmark.py           # Run benchmark suite
```

## Deployment and Setup

### 1. Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended for full LLM mode
- Network access for first-time model downloads (Qwen + sentence-transformers)

### 2. Install

```bash
git clone <your-repo-url> /opt/nexus-2
cd /opt/nexus-2

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Optional environment variables

```bash
# Faster/reliable model pulls from Hugging Face
export HF_TOKEN=<your_hf_token>

# Enables Gemini-based fact extraction in Info Learner paths
export GEMINI_API_KEY=<your_gemini_key>

# Pin GPU if needed
export CUDA_VISIBLE_DEVICES=0
```

### 4. Start modes

```bash
# Interactive CLI
python main.py

# One-shot non-interactive (JSON output)
python main.py --message "Hello"

# Persistent HTTP server (recommended for production integration)
python server.py --host 127.0.0.1 --port 8083
```

### 5. Verify server

```bash
curl -s http://127.0.0.1:8083/health
curl -s -X POST http://127.0.0.1:8083/interact \
  -H "Content-Type: application/json" \
  -d '{"message":"What is your name?"}'
```

### 6. Persistence paths (important)

Runtime memory is stored under:

- `data/memory/nexus2_memory.json`
- `data/memory/nexus2_memory.json.pt`

If you deploy with a process manager (systemd, docker, supervisor), keep a stable working directory (`WorkingDirectory=/opt/nexus-2`) so relative paths resolve correctly.

### 7. Systemd example

```ini
[Unit]
Description=NEXUS-2 Agent Server
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/nexus-2
Environment=PYTHONUNBUFFERED=1
Environment=HF_TOKEN=your_token_here
Environment=GEMINI_API_KEY=your_key_here
ExecStart=/opt/nexus-2/.venv/bin/python /opt/nexus-2/server.py --host 127.0.0.1 --port 8083
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 8. Common deployment pitfalls

- First startup can take time due to model downloads.
- If server returns empty memory behavior unexpectedly, confirm process working directory and memory files under `data/memory/`.
- Keep `--host 127.0.0.1` unless you explicitly intend external network exposure.

## Benchmarks

Four benchmark suites evaluate the system against RAG and LLM-only baselines:

- **Memory Recall** -- Seed k facts with distractors, query for specific attributes
- **Multi-hop Chain** -- 2-5 hop reasoning chains at various fact counts
- **Scalability** -- Performance at k=10, 50, 200, 500
- **vs RAG** -- Direct comparison with ChromaDB retrieval-augmented generation
