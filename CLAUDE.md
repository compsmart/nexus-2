# Nexus-2 Developer Agent Instructions

## Mission

Nexus-2 must work on REAL-WORLD text data — messy, ambiguous, unstructured natural language. Every change you make must be evaluated against that standard. If a technique only works on clean synthetic benchmark data, it is worthless. Do not ship it.

## Architecture

- **AdaptiveModularMemory** (`nexus2/memory/amm.py`): Neural text-to-embedding memory with SentenceTransformer encoder
- **InputProcessor** (`nexus2/perception/`): Perception pipeline
- **ChainExecutor** (`nexus2/reasoning/`): Multi-hop reasoning chains
- **LLMEngine + SoftPromptAdapter** (`nexus2/generation/`): Response generation with soft prompts
- **ToolRegistry + SkillStore** (`nexus2/action/`): Tool use and learned skills
- **ConsolidationLoop** (`nexus2/learning/`): Memory consolidation

## Non-Negotiable Engineering Standards

### 1. Real-World Robustness is Mandatory
- **NO hardcoded structural assumptions** about input text format
- All memory operations must handle: ambiguous entities, pronouns, implicit relationships, noisy/incomplete text, long documents
- Entity extraction must use proper NER or semantic methods — never string splitting or position-based heuristics
- Test mentally: "Would this work on a Wikipedia paragraph? A customer support transcript? A legal document?" If no, don't do it.

### 2. Benchmarks are Diagnostics, NOT Targets
- Benchmark scores measure progress — they are not the goal
- NEVER tune logic to match synthetic benchmark data patterns
- NEVER add special-case handling that only helps benchmark-shaped inputs
- If a change improves benchmark scores but relies on synthetic data structure, REJECT it
- A smaller benchmark improvement that generalizes is worth more than a large one that doesn't

### 3. No Shortcuts
- No assumptions about fact structure, ordering, or formatting in stored memories
- No prompt engineering hacks to dodge scorer edge cases
- No benchmark-specific optimizations that wouldn't apply to real heterogeneous data
- No gaming the evaluation — if the agent gets something wrong, fix the capability, not the output format

### 4. Implementation Quality
- Adapter integrity: < 60 lines, no `import re`, one call to real agent
- Changes go in agent code under `nexus2/` — NOT adapter.py
- Every improvement must degrade gracefully: if a technique fails, fall back to the simpler approach, not crash or return nothing
- The AMM, reasoning chain, and perception pipeline must handle messy real-world inputs at every stage

### 5. Neural Components Must Generalize
- Encoder improvements must be validated on diverse text, not just benchmark-formatted facts
- Memory bank retrieval must handle semantic similarity across paraphrases and linguistic variation
- Soft prompt tuning must not overfit to benchmark question patterns
- Consolidation must work when memories are heterogeneous in length, topic, and structure

## What Good Changes Look Like

- Improving AMM retrieval to handle paraphrased queries (same meaning, different words)
- Better chain reasoning that works when intermediate facts use pronouns or indirect references
- Perception improvements that extract entities from natural prose, not just structured facts
- Robustness: handling coreference, partial matches, ambiguous entities, multilingual input
- Making consolidation smarter about merging related but differently-worded memories

## What Bad Changes Look Like

- Any logic that assumes facts follow a specific template or format
- Tuning soft prompts specifically to benchmark question styles
- String-matching or position-based heuristics for entity identification
- Changing output phrasing to game scorer detection patterns
- Optimizing for the specific distractor/chain ratio in benchmarks

## Evaluation Criteria

When you benchmark a change, ask yourself:
1. Does this improvement come from better understanding of language, or from matching synthetic patterns?
2. Would this work if I fed in real conversation transcripts instead of synthetic facts?
3. Am I making the agent smarter, or just making it better at this specific test?

If the answer to #1 is "synthetic patterns" or #2 is "no" or #3 is "this specific test" — roll it back and try again.
