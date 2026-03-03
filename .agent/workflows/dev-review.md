---
description: Quick dev review — run tests, check for stubs, verify imports
---

# Dev Review Workflow

// turbo-all

1. Run the full test suite:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && OPENBLAS_NUM_THREADS=1 python3 -m pytest plonk/ crypto/ network/cnn/ -v --tb=short 2>&1 | tail -20
```

2. Scan for remaining stubs in proof path:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && grep -rn "return True  #\|return False  #\|raise NotImplementedError\|pass$" --include="*.py" plonk/ crypto/ | grep -v __pycache__ | grep -v test_
```

3. Check for _legacy shim usage:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && grep -rn "from _legacy" --include="*.py" . | grep -v __pycache__
```

4. Verify no German comments in key files:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && grep -rn "Prüfe\|Erstell\|Berechn\|vereinfacht\|Verifizier" --include="*.py" plonk/ crypto/ | grep -v __pycache__
```
