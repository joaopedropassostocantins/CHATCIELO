# CLAUDE.md — CHATCIELO Project Directives

> This file governs Claude's behavior in this repository.
> CHATCIELO is a **mission-critical system** for Cielo Brasil, implementing
> Pairwise Preference Ranking models (LMSYS SOTA) to personalize merchant
> support (MEI, Varejo, Corporate). Every decision must prioritize
> **quality, test coverage, and security**.

---

## 1. Build & Environment

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

All configuration is loaded from a `.env` file at the project root.
**Never hardcode credentials or PII-sensitive values.**

```bash
cp .env.example .env
# Edit .env with the appropriate values before running anything
```

Required variables (see `.env.example`):

```
MODEL_PATH=
API_HOST=
API_PORT=
LOG_LEVEL=
DB_URL=
```

---

## 2. Execution Commands

| Task | Command |
|---|---|
| Install dependencies | `pip install -r requirements.txt` |
| Train model | `python src/training/train.py` |
| Run API (dev) | `uvicorn src.api.main:app --reload` |
| Run API (prod) | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4` |
| Run all tests | `pytest tests/ -v` |
| Run with coverage | `pytest tests/ --cov=src --cov-report=term-missing` |
| Lint | `ruff check . && black --check .` |
| Format | `black .` |
| Type check | `mypy src/` |
| Benchmarks | `pytest tests/benchmarks/ --benchmark-only` |

---

## 3. Testing Protocol — NON-NEGOTIABLE

> **MANDATORY RULE**: Claude never delivers code without the corresponding test.
> When a feature is requested, the **test is written and shown first**, then the implementation.

### 3.1 Unit Tests

- Every new component under `src/` **must** have a corresponding test in `tests/unit/`.
- Framework: **pytest**
- Naming convention: `tests/unit/test_<module_name>.py`
- Minimum coverage threshold: **80% per module** (enforced via `pytest --cov`).

```bash
pytest tests/unit/ -v
```

### 3.2 Integration Tests

Validate the full end-to-end flow:

```
Dataloader → Model → Scoring
```

- Located in `tests/integration/`
- Must assert correct data shapes, types, and score ranges across the full pipeline.
- Must test both happy-path and failure modes (empty input, malformed records).

```bash
pytest tests/integration/ -v
```

### 3.3 Performance Benchmarking

- Framework: **pytest-benchmark**
- Located in `tests/benchmarks/`
- **Acceptance criterion: P99 latency < 300ms** for the inference endpoint.
- If a new change causes P99 to exceed 300ms, the code **must be optimized before merging**.

```bash
pytest tests/benchmarks/ --benchmark-only --benchmark-histogram
```

Benchmark assertions example:

```python
def test_inference_latency(benchmark):
    result = benchmark(run_inference, sample_input)
    assert benchmark.stats["max"] < 0.300  # 300ms hard ceiling
```

### 3.4 ML Model Tests

Property-based testing with **hypothesis** to guarantee model correctness:

- Preference scores must always be in the range **[0, 1]**.
- The model must be **resilient to malformed inputs** (missing fields, null values, unexpected types).
- Located in `tests/ml/`

```bash
pytest tests/ml/ -v
```

Example:

```python
from hypothesis import given, strategies as st

@given(st.floats(allow_nan=True, allow_infinity=True))
def test_score_range(raw_score):
    score = normalize_score(raw_score)
    assert 0.0 <= score <= 1.0
```

### 3.5 Security & LGPD Audit

Scripts to detect **PII in logs and outputs** (CPF, card numbers, names) are
located in `tests/security/`.

- These tests **block any commit** that leaks PII to logs or API responses.
- PII patterns to detect: CPF (`\d{3}\.\d{3}\.\d{3}-\d{2}`), card numbers (Luhn-valid 13–19 digit sequences), full names in structured outputs.

```bash
pytest tests/security/ -v
```

Audit script (run manually or in CI):

```bash
python scripts/lgpd_audit.py --target logs/ --target outputs/
```

---

## 4. Code Style & Standards

### Linting & Formatting

Both checks **must pass before any commit**:

```bash
ruff check .
black .
```

Configure pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

`.pre-commit-config.yaml` must include `ruff`, `black`, and `mypy`.

### Type Safety

**mypy is mandatory** for all modules under `src/`:

```bash
mypy src/ --strict
```

No `# type: ignore` comment is accepted without a written justification in the same line.

### Documentation

**Google Style Docstrings** are required for every function and class.
The docstring must document:

- `Args`: all parameters with types and descriptions.
- `Returns`: return type and semantic meaning.
- `Raises`: all exceptions that may be raised.
- **Validation Metrics**: for ML functions, document the validation metric used
  (e.g., accuracy, AUC, preference score range).

Example:

```python
def compute_preference_score(logits: torch.Tensor) -> float:
    """Converts raw model logits to a normalized preference score.

    Args:
        logits: Raw output tensor from the ranking model, shape (1, 2).

    Returns:
        A float in [0, 1] representing the preference probability for
        response A over response B. Validated via softmax invariant.

    Raises:
        ValueError: If logits shape is not (1, 2).

    Validation Metrics:
        - Score range: must satisfy 0.0 <= score <= 1.0.
        - Calibration: ECE < 0.05 on held-out eval set.
    """
```

---

## 5. Critical Technical Context

### `src/data/dataset.py` — Heart of Context Injection

> **Any modification to `src/data/dataset.py` requires full re-validation
> of the training data pipeline.**

Checklist before merging changes to this file:

- [ ] Re-run `tests/integration/` in full.
- [ ] Verify tokenization output shapes are unchanged.
- [ ] Confirm preference label distribution has not shifted (run `scripts/validate_dataset.py`).
- [ ] Update docstrings to reflect any schema changes.

### Horizontal Scalability

The system is designed to be **stateless**:

- No in-process state between requests.
- Session/context information must be passed explicitly per request.
- Any new component that introduces server-side state **must be rejected** or
  moved to an external store (Redis, DB) with a justification comment.

---

## 6. Claude Behavioral Rules (Hard Rules)

1. **Test first, code second.** When asked for a feature, deliver the test file before the implementation.
2. **No untested code.** If a code snippet is provided without a test, flag it immediately and write the test.
3. **No PII in examples.** Never use real CPF, card numbers, or names in code samples, docstrings, or test fixtures. Use `Faker` or clearly synthetic data.
4. **Latency awareness.** For any inference-path change, explicitly reason about P99 latency impact before suggesting the implementation.
5. **`dataset.py` is a blast radius.** Always warn when a change may touch the data pipeline and require re-validation.
6. **LGPD by default.** When logging is involved, always add PII-scrubbing logic. Never log raw user input without sanitization.
7. **Stateless by default.** Never introduce server-side state without explicit discussion.
8. **Type everything.** All new functions must have full type annotations. No `Any` without justification.
