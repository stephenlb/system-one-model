# Comparison: this repo (System One / Jev replica) vs Strands Decider 2B

Source: https://strandsagents.com/blog/introducing-strands-decider/
Run date 2026-10-01, Apple Silicon Mac (64 GB, MPS), bf16, transformers 5.17/5.18.

## Design comparison

| | This repo (`truetype`) | Strands Decider 2B (v19) |
|---|---|---|
| Base | `google/gemma-4-12B` (~22-24 GB) | Qwen3.5-2B torso + LoRA r16 (~2B) |
| Output head | LM head replaced by 26 A-Z logits (one logit per answer slot) | ~1M-param "pointer head": compares hidden state at each option position to the `<answer>` position |
| Training | None: prompt-engineered (few-shot roundtrip demos, state last); frozen weights | Fine-tuned LoRA on decision data; training data/scripts shipped |
| Question types | `noul`, `choice` (<=26), `score` (ordered levels, <=26) | `noul`, `choice` (<=255), `score` (2-10 levels) |
| API | `/v1/systemone`, HF `system-one` pipeline, `TrueTypeClient` | Same `/v1/systemone` schema, `strands-decider ask/serve` CLI, pip package |
| Score output | expected level `sum(i*p_i)` + confidence | same formulation, plus `legend` |
| Confidence | `1 - H/log n` | derived from probabilities (same idea) |
| Multi-question | batched, prefix KV cache | one parallel pass, shared prefix |
| Extras | Doom/Mario/Flappy/Tetris demos, HF model, Space | Strands agent `InterventionHandler` integration, JevBench eval, training code |
| Hardware | ~24 GB RAM minimum | runs on CPU / consumer GPU |

Key trade-off: we get strong zero-training accuracy from a much bigger model but pay 10x in memory and load time; Strands is small, trained specifically for decisions, and better integrated with agent frameworks.

## Local test results

**Our test suite** (`pytest tests`, 12B model on MPS): **141 passed in 37.6 s**.

**Head-to-head** (`cases.py`: 15 noul, 12 choice, 12 score, identical inputs via the shared `/v1/systemone` schema; `run_bench.py`, `report.py`; raw data in `results_*.json`):

| metric | ours (Gemma-4 12B, local) | Strands Decider 2B v19 (local) | TypeSafe Jev API (hosted) |
|---|---|---|---|
| noul acc (/15) | 15 | 15 | 15 |
| noul Brier (lower better) | 0.0 | 0.03379 | 0.00035 |
| choice acc (/12) | 11 | 12 | 12 |
| choice mean P(correct) | 0.924 | 0.964 | 0.993 |
| score round-acc (/12) | 11 | 5 | 12 |
| score pairwise ordering | 1.0 | 0.958 | 1.0 |
| score MAE (levels) | 0.09 | 0.483 | 0.004 |
| load s | 14.8 | 45.8 | 0.0 |
| 1-q p50 ms | 117 | 96 | 144 |
| 1-q p95 ms | 473 | 190 | 207 |
| 3-q p50 ms | 332 | 121 | 164 |

### Summary (strict rounded accuracy over 39 cases): Jev API 39/39, ours 37/39, Strands 32/39

The hosted Jev API leads every accuracy metric. Among local models, accuracy is a near tie; Strands wins on latency and footprint, ours wins on `score` calibration and sharpness.

### Caveats (read before quoting)
- Small, hand-written sample (39 cases); differences of 1 item are noise. Our choice/score prompts were similar to cases used while developing this repo (bakeoff tests), so our numbers may be optimistic. Not JevBench.
- Strands' score outputs are compressed (0.68-1.6 for true levels 0-2). Rounding penalises it (5/12), but the ordering is nearly right (0.958), so it is usable with a tuned threshold, not raw rounding.
- Our one choice miss: "refund has not shown up" -> billing; our one score miss: "dark mode unreadable but can switch" -> 0.24.
- Latency: single run on one machine, timings vary (our p95 includes some slow first-seen prompts). The blog's 115 ms (3090) / 153 ms (M3) figures are consistent with our 96 ms.
- Strands' public `strands-decider 0.1.0` has no `Decider.noul(...)` class as shown in the blog; we used `strands_decider.infer.load_engine` / `SystemOneRequest`, the same path the CLI uses. Its `causal_conv1d` fallback kernel is slower on MPS; a tuned install may be faster.
- Did not run Strands' JevBench or agent-intervention example (needs Bedrock credentials).

### Hosted TypeSafe Jev API (`jev-latest`)
`run_bench.py typesafe` calls `POST https://api.typesafe.ai/v1/systemone` with `TYPESAFE_API_KEY` (per https://docs.typesafe.ai/introduction/quickstart). Its latency includes the network round trip from this machine, so it is not comparable to on-device numbers; it is the only system that needs no local model. Notes: the API returns HTTP 400 if the request carries a `temperature` field (our own `TrueTypeClient` sends one, so the benchmark uses a plain request); the python.org Python here needed `SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')`.

## Reproduce
```bash
# ours (repo .venv)
cd .. && .venv/bin/python -m pytest tests -q
.venv/bin/python comparison-strands-decider/run_bench.py ours   # run from repo root; move results_ours.json here
# strands
cd comparison-strands-decider && python3 -m venv .venv-strands && .venv-strands/bin/pip install strands-decider
.venv-strands/bin/python run_bench.py strands
SSL_CERT_FILE=$(.venv-strands/bin/python -c 'import certifi;print(certifi.where())') \
  ../.venv/bin/python run_bench.py typesafe      # needs TYPESAFE_API_KEY
python3 report.py
```
