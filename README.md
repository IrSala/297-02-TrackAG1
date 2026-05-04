# 297-02-TrackAG1 — Qwen LLM Governance Audit

**CMPE 297-02 | San José State University | Spring 2026**  
**Team:** Jimmy Chung · Harshit Jangam · Rajitha Muthukrishnan · Irwin Salamanca · Ritesh Rakesh Singh

---

## Overview

This repository contains the empirical probe scripts, results, and datasets for our governance audit and threat mapping of the **Qwen 2.5 (7B)** and **Qwen 3.5 (9B)** large language models developed by Alibaba Cloud.

All tests were conducted **offline via local Ollama** — no live APIs or external systems were contacted during probing.

**MITRE ATLAS Framework:** All risks are mapped to ATLAS TTPs.  
**Models tested:** `qwen2.5:7b` and `qwen3.5:9b` via Ollama.

---

## Running the Probes

Make sure `ollama serve` is running before executing any script.

```bash
# Risk 1 — Prompt Injection
python scripts/R1.py

# Risk 3 — Capability Probing
python scripts/R3.py

# Risk 4 — Resource Exhaustion
python scripts/R4.py

# Risk 5 — Model Modification
python scripts/R5.py

# Risk 6 — Training Data Memorization
python scripts/R6.py

# Risk 7 — Domain Safety Weakness
python scripts/R7.py

# Risk 8 — Bias (requires StereoSet cache)
python scripts/R8.py

# Risk 9 — Multi-Step Attack Chaining
python scripts/R9.py

# Risk 10 — Instruction/Data Confusion (full probe, both models)
python scripts/R10.py

# Risk 10 — Qwen 3.5 rerun only (extended 300s timeout)
python scripts/R10.py --rerun

# Risk 11 — Overreliance (requires TruthfulQA CSV)
python scripts/R11.py
```

Each script outputs timestamped result files (CSV, JSONL, and/or TXT) to the working directory. These are excluded from the repo by `.gitignore`.

---

## Notes

- All probes are **offline only** — no live systems or external APIs are tested
- Verdict labels are auto-heuristic; manual review is recommended, especially for R6, R8, and R10
- Qwen 3.5 uses visible chain-of-thought reasoning which significantly increases response time — use extended timeouts where noted
- R5 performs a non-destructive tamper simulation: original model files are never permanently modified
