---
description: Iterative OpenAI co-review audit — file-by-file proof path analysis
---

# Audit Loop Workflow

For each module in the audit scope:

1. **Select file** from the audit queue (crypto → polynomial → KZG → PLONK → network)

2. **Send to OpenAI for review:**
   Use `mcp_openai_review_code_with_openai` with focus areas:
   - Soundness bugs (anything that silently accepts invalid proofs)
   - Hollow shells (pass stubs, return True without logic)
   - Missing edge-case handling
   - Field/curve consistency

3. **Triage findings** by severity:
   - CRITICAL: Fix immediately before proceeding
   - HIGH: Fix in current iteration
   - MEDIUM: Log to task.md for next iteration
   - LOW: Document only

4. **Apply fixes** and re-run tests:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && OPENBLAS_NUM_THREADS=1 python3 -m pytest plonk/ crypto/ network/cnn/ -v --tb=short 2>&1 | tail -10
```

5. **Repeat** until all files in scope are audited

6. **Commit** after each complete audit pass:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && git add -A && git status --short
```
