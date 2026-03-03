---
description: Pre-release validation — comprehensive checks before tagging
---

# Release Check Workflow

// turbo-all

1. Full test suite (must be 100% pass):
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && OPENBLAS_NUM_THREADS=1 python3 -m pytest plonk/ crypto/ network/cnn/ -v --tb=short 2>&1
```

2. No _legacy shims remaining:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && grep -rn "from _legacy" --include="*.py" . | grep -v __pycache__ && echo "FAIL: _legacy shims found" || echo "PASS: no _legacy shims"
```

3. No hardcoded tau/secrets:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && grep -rn "Fr(12345\|tau = Fr(" --include="*.py" . | grep -v __pycache__ | grep -v _legacy | grep -v "test_" | grep -v "generate_insecure" && echo "FAIL: hardcoded secrets" || echo "PASS: no hardcoded secrets"
```

4. No German comments in core:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && count=$(grep -rcn "Prüfe\|Erstell\|Berechn\|vereinfacht\|Verifizier\|öffentlich\|Eingab\|zugehörig" --include="*.py" plonk/ crypto/ network/ | grep -v ":0$" | wc -l) && echo "German comment files: $count" && [ "$count" -eq 0 ] && echo "PASS" || echo "WARN: German remaining"
```

5. No return-True stubs in proof path:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && grep -rn "return True  # Placeholder\|return True  # Stub\|return True  # TODO" --include="*.py" plonk/ crypto/ | grep -v __pycache__ && echo "FAIL: stubs found" || echo "PASS: no stubs"
```

6. LICENSE exists:
```bash
test -f /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system/LICENSE && echo "PASS: LICENSE exists" || echo "FAIL: no LICENSE"
```

7. README has working example:
```bash
head -5 /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system/README.md
```

8. Git status clean:
```bash
cd /home/dweyh/franklin/llm_fingerprint/tda_review/zkml_review/zkml_system && git status --short
```
