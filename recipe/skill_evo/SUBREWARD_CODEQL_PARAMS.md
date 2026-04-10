# CodeQL Subreward Parameters

This note explains the CodeQL-related subreward knobs used by `run_grpo.sh`.

## Reward Logic

CodeQL subreward is applied **only when all tests pass** (`passed == total` and `total > 0`).

Formula:

```text
raw_scalar = clip01(0.8 * (1 - exp(-density)) + 0.2 * (1 - exp(-nodes / 120)))
density = edges / nodes

robust_score = clip01((raw_scalar - threshold) / scale)
subreward = weight * robust_score
final_reward = clip01(base_reward + subreward)
```

## Parameters

### `DEEPCODER_ENABLE_CODEQL_SUBREWARD` (default: `true`)
- Turns CodeQL subreward on/off.
- `true`: enable subreward.
- `false`: only base reward.

### `DEEPCODER_CODEQL_SUBREWARD_WEIGHT` (default: `0.10`)
- Weight of subreward added to base reward.
- Larger value gives stronger optimization pressure toward CodeQL robustness.

### `DEEPCODER_CODEQL_SUBREWARD_THRESHOLD` (default: `0.82`)
- Baseline for `raw_scalar`.
- If `raw_scalar <= threshold`, normalized robustness reward tends to 0.

### `DEEPCODER_CODEQL_SUBREWARD_SCALE` (default: `0.15`)
- Controls how fast `robust_score` increases above threshold.
- Smaller scale: more sensitive.
- Larger scale: smoother, less sensitive.

### `DEEPCODER_CODEQL_REQUIRE_OK` (default: `true`)
- Whether CodeQL analysis must succeed before giving subreward.
- `true`: if CodeQL fails, subreward = 0.
- `false`: still compute from returned scalar (usually conservative fallback).

### `DEEPCODER_CODEQL_TIMEOUT_S` (default: `120`)
- Timeout for each CodeQL command in reward-time analysis.
- Increase if frequent timeout on longer samples.

### `DEEPCODER_CODEQL_BIN` (default: empty)
- Path to CodeQL executable, e.g.:
  - `$(command -v codeql)`
- If empty, code tries `codeql` from `PATH`.

### `DEEPCODER_CODEQL_WORKDIR` (default: empty)
- Parent directory for temporary CodeQL working folders.
- If empty, system temp dir is used.

## Recommended Starting Setup

```bash
export DEEPCODER_ENABLE_CODEQL_SUBREWARD=true
export DEEPCODER_CODEQL_SUBREWARD_WEIGHT=0.10
export DEEPCODER_CODEQL_SUBREWARD_THRESHOLD=0.82
export DEEPCODER_CODEQL_SUBREWARD_SCALE=0.15
export DEEPCODER_CODEQL_REQUIRE_OK=true
export DEEPCODER_CODEQL_TIMEOUT_S=120
export DEEPCODER_CODEQL_BIN="$(command -v codeql)"
# If command -v returns empty, set it explicitly:
# export DEEPCODER_CODEQL_BIN="<path-to-your-codeql-binary>"
export DEEPCODER_CODEQL_WORKDIR="./codeql_tmp"
```
