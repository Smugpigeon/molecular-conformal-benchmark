#!/usr/bin/env bash
# JCIM reproducibility checklist. Per CLAUDE.md §13.
# Run before submitting paper draft.

set -uo pipefail
PASS=0
FAIL=0

pass() { echo "  [PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "JCIM Reproducibility Checklist"
echo "================================"
echo

echo "[1] Environment lock files"
[[ -f env.yml ]]              && pass "env.yml exists"                || fail "env.yml missing"
[[ -f pyproject.toml ]]       && pass "pyproject.toml exists"         || fail "pyproject.toml missing"
echo

echo "[2] Seed management"
grep -q "set_all_seeds" src/train.py && pass "train.py calls set_all_seeds" \
                                      || fail "train.py missing set_all_seeds"
grep -q "set_all_seeds" src/evaluate.py && pass "evaluate.py calls set_all_seeds" \
                                         || fail "evaluate.py missing set_all_seeds"
echo

echo "[3] Test/val separation"
grep -q -- "--final" src/evaluate.py && pass "evaluate.py guards test split with --final" \
                                     || fail "evaluate.py missing --final guard"
echo

echo "[4] Data immutability"
if [[ -d data/processed ]]; then
    pass "data/processed/ exists (regeneratable)"
else
    echo "  [INFO] data/processed/ not yet built; run 'make data'"
fi
[[ -f Makefile ]] && grep -q "^canonicalize" Makefile \
                  && pass "make canonicalize exists" \
                  || fail "make canonicalize target missing"
echo

echo "[5] Documentation"
[[ -f README.md ]] && pass "README.md exists"  || fail "README.md missing"
[[ -f CLAUDE.md ]] && pass "CLAUDE.md exists"  || fail "CLAUDE.md missing"
grep -q "Quick Start" README.md && pass "README has Quick Start"  || fail "README missing Quick Start"
echo

echo "[6] Tests"
if command -v pytest >/dev/null 2>&1; then
    pytest -x -q tests/ >/tmp/pytest.log 2>&1 \
        && pass "pytest passes" \
        || { fail "pytest failures (see /tmp/pytest.log)"; tail -20 /tmp/pytest.log; }
else
    echo "  [INFO] pytest not installed; skipping test run"
fi
echo

echo "[7] Citations / references"
grep -q "10.1021/acs.jcim.4c01591" configs/dataset/gsht.yaml \
    && pass "GSHt source paper (Zhang JCIM 2025) cited in config" \
    || fail "GSHt config missing Zhang 2025 DOI"
echo

echo "================================"
echo "Result: $PASS pass, $FAIL fail"
[[ $FAIL -eq 0 ]] || exit 1
