# DeepCoder + SandboxFusion deployment on this host

## What you need to be able to run DeepCoder experiments

1. SandboxFusion service running locally (executes model-generated code against test cases).
2. DeepCoder dataset in verl parquet format.
3. A text-only Qwen3-4B (or larger) model in the local HF cache.
4. The recipe scripts under `recipe/pdpo/run_*_deepcoder_*.sh`.

This doc walks through all four. Everything except the local file paths is reproducible.

---

## 1. SandboxFusion

### Install

```bash
# Repo + service env
cd /workspace
git clone --depth 1 https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion

# Server env (Python 3.11 + FastAPI etc.) — keep it OUT of the verl env.
conda create -n sandbox-service -y python=3.11
/root/miniconda3/envs/sandbox-service/bin/pip install --quiet \
    "fastapi==0.103.*" "uvicorn[standard]==0.25.0" "pydantic>=2.4.0,<2.7.0" \
    structlog psutil aiofiles aiohttp tenacity "databases[aiosqlite]" \
    "transformers>=4.44.0" pyyaml python-dotenv datasets

# Runtime env (Python 3.10) — where user-submitted code actually runs.
# Minimal: stdlib + numpy is enough for DeepCoder competitive-programming
# problems. The upstream install-python-runtime.sh installs TF / PyTorch /
# Django etc., which DeepCoder doesn't need.
conda create -n sandbox-runtime -y python=3.10
/root/miniconda3/envs/sandbox-runtime/bin/pip install --quiet numpy

# Server expects docs/build to exist — create a stub if it doesn't.
mkdir -p docs/build
[ -f docs/build/index.html ] || echo "<html></html>" > docs/build/index.html
```

### Start

```bash
SBX_PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
echo "$SBX_PORT" > /workspace/SandboxFusion/.run/port

cd /workspace/SandboxFusion
nohup /root/miniconda3/envs/sandbox-service/bin/uvicorn \
    sandbox.server.server:app --host 127.0.0.1 --port "$SBX_PORT" \
    > .run/sandbox.log 2>&1 < /dev/null &
echo $! > .run/sandbox.pid
disown
```

### Verify

```bash
SBX_PORT=$(cat /workspace/SandboxFusion/.run/port)
curl -s -X POST "http://127.0.0.1:${SBX_PORT}/run_code" \
    -H "Content-Type: application/json" \
    --data-raw '{"code": "print(sum(range(100)))", "language": "python"}'
# expected: {"status":"Success", ..., "stdout":"4950\n", ...}
```

### Stop

```bash
kill -9 $(cat /workspace/SandboxFusion/.run/sandbox.pid)
```

---

## 2. DeepCoder dataset

```bash
cd /workspace/PDPO
PYTHONPATH=/workspace/PDPO /root/miniconda3/envs/verl/bin/python \
    -m recipe.pdpo.data_preprocess.deepcoder \
    --output_dir /workspace/data/deepcoder \
    --train_subset taco --train_max 2000 \
    --val_subset codeforces --val_max 200
```

Source: HuggingFace `agentica-org/DeepCoder-Preview-Dataset`. Subsets:
- `taco` — 7.4K problems, only `train` split, used as the training source.
- `codeforces` — 408 problems, only `test` split, used as evaluation (recent contests Aug-2024 → Feb-2025, so unlikely to be in pretraining).
- `primeintellect` (16.3K), `lcbv5` (600) are also available, swap via `--train_subset`/`--val_subset`.

Output:
```
/workspace/data/deepcoder/train.parquet     # 2000 rows × {prompt, data_source, ability, reward_model, extra_info}
/workspace/data/deepcoder/test.parquet      # 200 rows, same schema
```

The `reward_model.ground_truth` is the JSON-string list of `{input, output}` test cases — read back by `recipe/pdpo/reward_score/deepcoder_action_thought_reward.py:_get_tests_deepcoder`.

---

## 3. Qwen3-4B

```bash
/root/miniconda3/envs/verl/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id='Qwen/Qwen3-4B',
    allow_patterns=['*.json', '*.txt', '*.safetensors', 'tokenizer*'],
))"
```

Resolves to:
```
/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
```

(36 layers, 32/8 GQA, 32K native ctx, text-only, instruct-tuned.)

---

## 4. Run

The training scripts are already wired:

```bash
# GRPO baseline (single primary reward)
CUDA_VISIBLE_DEVICES=7 bash recipe/pdpo/run_grpo_deepcoder_1gpu.sh

# PD-GDPO (primary + 2 AST aux, adaptive λ)
CUDA_VISIBLE_DEVICES=7 bash recipe/pdpo/run_pdpo_deepcoder_1gpu.sh
```

Both pick up the SandboxFusion port from `/workspace/SandboxFusion/.run/port` automatically. Set `SANDBOX_FUSION_URL` to override.

### Why the reward_kwargs look this way

`coding_enable_sub_rewards: true` turns on `collect_subrewards("coding", ctx)`. Two of the four heuristics need eval results we don't (currently) thread through to the reward worker:
- `coding_compiler_runtime_feedback` reads `ctx["eval_total/passed/error"]` — disabled.
- `coding_executed_token_credit` reads same, plus optional `ctx["executed_lines"]` — disabled.

Left enabled (pure-AST, no sandbox call):
- `coding_static_analysis_reward` — penalises `eval`/`exec`/lack of callable structure.
- `coding_block_level_process_reward` — fraction of code prefixes that parse.

PD-GDPO consumes these via `algorithm.pd_gdpo.component_keys=[coding_static_analysis_reward,coding_block_level_process_reward]`.

### Memory notes for single H200

`actor_rollout_ref.rollout.gpu_memory_utilization=0.35` carves out ~50 GB for vLLM KV; FSDP NO_SHARD on a single GPU keeps the Qwen3-4B + grad + optim around 30 GB; activations consume the rest. `enforce_eager=true` and `attn_implementation=eager` were necessary on this host to dodge the `std::bad_alloc` from torch.compile + NCCL CUMEM (see GSM8K experiment notes). The required env vars are exported inside each `run_*_1gpu.sh`:

```bash
export NCCL_CUMEM_ENABLE=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

If you skip those, vLLM crashes during KV-cache init with no useful error.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Sandbox `/run_code` times out | Check `/workspace/SandboxFusion/.run/sandbox.log` for python tracebacks. The sandbox-runtime conda env must exist (`conda env list` should show it). |
| `Failed to import Triton kernels ... gpt_oss` | Harmless. Triton mismatch only affects MoE kernels you don't use. |
| `Unknown advantage estimator pd_gdpo` | `recipe.pdpo` wasn't imported inside the Ray TaskRunner actor. Use `python -m recipe.pdpo.main_pdpo` (not `verl.trainer.main_ppo`). The custom TaskRunner imports the recipe inside the actor. |
| `KeyError 'pred'` | Your `compute_score` returns dicts with inconsistent key sets across samples. Always emit the same keys; pad missing ones with `""` or `0.0`. |
| `std::bad_alloc` during vLLM init | Set the five env vars above before launching. |

---

## Known SandboxFusion patch (required for stdin-driven problems)

Vanilla `sandbox/runners/base.py` calls `p.stdin.flush()`, but `asyncio.subprocess` exposes a `StreamWriter` whose proper coroutine-aware drain is `await p.stdin.drain()`. The vanilla code fails silently on every codeforces problem (which feeds tests via stdin), making the training run appear to hang.

Patch:

```python
# /workspace/SandboxFusion/sandbox/runners/base.py, around line 73
# REPLACE:
#     p.stdin.write(stdin.encode())
#     p.stdin.flush()
# WITH:
p.stdin.write(stdin.encode())
drain = getattr(p.stdin, 'drain', None)
if callable(drain):
    await drain()
```

After patching, restart the sandbox service. Verify with:

```bash
SBX_PORT=$(cat /workspace/SandboxFusion/.run/port)
curl -sf -X POST "http://127.0.0.1:${SBX_PORT}/run_code" \
    -H "Content-Type: application/json" \
    --data-raw '{"code":"import sys;print(sum(int(x) for x in sys.stdin.read().split()))","language":"python","stdin":"1 2 3\n"}'
# expected: stdout "6\n"
```
