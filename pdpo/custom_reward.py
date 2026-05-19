# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Component-emitting ``compute_score`` for PD-GDPO.

This function returns a dict whose top-level keys flow into
``reward_extra_info`` (verl's ``NaiveRewardManager`` lifts every key of a
dict-valued ``compute_score`` return into ``reward_extra_info``, and
``ray_trainer.py`` then promotes them into ``data.non_tensor_batch`` as
numpy arrays).  The PD-GDPO advantage estimator reads those arrays back
out.

Convention:
    * ``"score"`` -- the primary scalar reward (correctness for math,
      pass-rate for coding).  This is what ``token_level_scores`` gets
      populated with, and it is what the correctness gate is applied to.
    * ``"<category>_<name>"`` -- each auxiliary sub-reward, as emitted by
      :func:`collect_subrewards`.  These names match the
      ``algorithm.pd_gdpo.component_keys`` config field.
    * ``"acc"`` -- legacy convenience field, mirrors the boolean
      correctness of the primary reward.
    * ``"original_reward"`` -- the raw primary score before any
      transformations (useful for logging).

No scalarisation happens here -- λ_k lives only inside the advantage
estimator.  This is the deliberate split that distinguishes PD-GDPO from
"scalar PD".
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any


# Module-level concurrency cap for the deepcoder/sandbox path. Without
# this, verl's check_correctness fires up to ~800 concurrent threads per
# (problem × generation) call, and verl's agent_loop blocks in ep_poll
# waiting for the slowest call. Each Ray reward worker process gets its
# own copy of this semaphore; verl typically spawns ~8 reward workers,
# so total concurrent sandbox calls ≈ 8 × DEEPCODER_SANDBOX_MAX_PARALLEL.
_DEEPCODER_SANDBOX_MAX_PARALLEL = int(os.environ.get("DEEPCODER_SANDBOX_MAX_PARALLEL", "16"))
_DEEPCODER_SANDBOX_SEMAPHORE = threading.Semaphore(_DEEPCODER_SANDBOX_MAX_PARALLEL)

from verl.utils.reward_score import default_compute_score, math_dapo

# Make local sub_reward package importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from reward_score import collect_subrewards  # noqa: E402  -- relies on sys.path
from reward_score.sub_reward.common import to_bool  # noqa: E402


_MATH_SOURCES = {
    "math",
    "math_dapo",
    "math_dapo_reasoning",
    "openr1_math_220k",
    "general365",
    "deepscalar",
    "gsm8k",
    "openai/gsm8k",
}


def _is_math(data_source: Any) -> bool:
    ds = str(data_source)
    if ds in _MATH_SOURCES:
        return True
    return ds.startswith("aime") or ds.startswith("deepscalar")


_CODING_SOURCES = {"mbpp", "mbpp:train", "mbpp:test", "mbpp:validation"}


def _is_coding(data_source: Any) -> bool:
    ds = str(data_source)
    if ds in _CODING_SOURCES:
        return True
    return ds.startswith("deepcoder") or ds.startswith("mbpp")


def _coerce_primary(base_res: Any) -> tuple[float, bool]:
    if isinstance(base_res, dict):
        score = float(base_res.get("score", base_res.get("acc", 0.0)))
        acc = bool(base_res.get("acc", score > 0.0))
        return score, acc
    score = float(base_res)
    return score, score > 0.0


def _score_math(data_source, solution_str, ground_truth, extra_info, kwargs):
    if str(data_source) in {"gsm8k", "openai/gsm8k"}:
        from verl.utils.reward_score import gsm8k

        base_res = gsm8k.compute_score(solution_str, ground_truth)
        if (isinstance(base_res, dict) and not base_res.get("acc", False)) or (
            isinstance(base_res, (int, float)) and float(base_res) == 0.0
        ):
            if "\\boxed{" in (solution_str or ""):
                base_res = math_dapo.compute_score(solution_str, ground_truth)
            else:
                base_res = gsm8k.compute_score(solution_str, ground_truth, method="flexible")
    else:
        base_res = math_dapo.compute_score(solution_str, ground_truth)

    primary, acc = _coerce_primary(base_res)

    ctx = {
        "response": solution_str,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info or {},
        "data_source": data_source,
        "base_score": primary,
        "base_acc": acc,
    }
    subrewards = collect_subrewards("math", ctx, **kwargs) if kwargs else {}

    # Always emit the same key set per sample. verl's agent loop
    # (verl/experimental/agent_loop/agent_loop.py:949) builds a
    # non_tensor_batch by collecting reward_extra_info keys across the
    # batch and indexing every sample by those keys; if any key is
    # present in some samples and absent in others, it raises KeyError.
    pred_value = ""
    if isinstance(base_res, dict) and "pred" in base_res:
        pred_value = base_res["pred"]

    info: dict[str, Any] = {
        "score": float(primary),
        "acc": bool(acc),
        "original_reward": float(primary),
        "pred": pred_value,
    }
    for name, value in subrewards.items():
        info[name] = float(value)
    return info


def _score_coding(data_source, solution_str, ground_truth, extra_info, kwargs):
    # For DeepCoder (and friends), primary correctness comes from running
    # the model output against test cases inside SandboxFusion. The
    # deepcoder reward function returns (main_reward, internal_subrewards);
    # we discard the internal subrewards and instead compute the
    # heuristic AST-based aux signals via collect_subrewards("coding", ...),
    # so the experiment matches the math case (heuristic aux on top of a
    # primary correctness signal).
    if str(data_source).startswith("deepcoder"):
        from reward_score import deepcoder_action_thought_reward as _dc

        sandbox_url = kwargs.get("sandbox_fusion_url") or kwargs.get("sandbox_url")
        # Use the module-level semaphore unless the caller explicitly passed one.
        concurrent_semaphore = kwargs.get("concurrent_semaphore", _DEEPCODER_SANDBOX_SEMAPHORE)
        res = _dc.compute_score_deepcoder(
            solution_str,
            ground_truth=ground_truth,
            sandbox_url=sandbox_url,
            concurrent_semaphore=concurrent_semaphore,
            return_components=True,
            **{k: v for k, v in kwargs.items() if k not in {"sandbox_fusion_url", "sandbox_url", "concurrent_semaphore"}},
        )
        if isinstance(res, tuple) and len(res) == 2:
            primary = float(res[0])
            base_acc = primary > 0.0
        elif isinstance(res, dict):
            primary = float(res.get("score", res.get("main_reward", 0.0)))
            base_acc = bool(res.get("acc", primary > 0.0))
        else:
            primary = float(res)
            base_acc = primary > 0.0
    elif str(data_source).startswith("mbpp"):
        # MBPP scorer runs unit tests in a local subprocess (no sandbox
        # needed). The reward function's _get_tests handles either a
        # Python list or a newline-joined string; if we got a JSON-encoded
        # list string from the dataset, decode it first.
        from reward_score import mbpp_action_thought_reward as _mbpp

        gt = ground_truth
        if isinstance(gt, str):
            stripped = gt.strip()
            if stripped.startswith("["):
                try:
                    decoded = json.loads(stripped)
                    if isinstance(decoded, list):
                        gt = {"test_list": [str(x) for x in decoded]}
                except json.JSONDecodeError:
                    pass

        res = _mbpp.compute_score_mbpp(
            solution_str,
            ground_truth=gt,
            return_components=True,
            **{k: v for k, v in kwargs.items() if k not in {"sandbox_fusion_url", "sandbox_url", "concurrent_semaphore"}},
        )
        if isinstance(res, tuple) and len(res) == 2:
            primary = float(res[0])
            base_acc = primary > 0.0
        elif isinstance(res, dict):
            primary = float(res.get("score", res.get("main_reward", 0.0)))
            base_acc = bool(res.get("acc", primary > 0.0))
        else:
            primary = float(res)
            base_acc = primary > 0.0
    else:
        base_res = default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
        primary, base_acc = _coerce_primary(base_res)

    ctx = {
        "response": solution_str,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info or {},
        "data_source": data_source,
        "base_score": primary,
        "base_acc": base_acc,
    }
    subrewards = collect_subrewards("coding", ctx, **kwargs) if kwargs else {}
    info: dict[str, Any] = {
        "score": float(primary),
        "acc": bool(base_acc),
        "original_reward": float(primary),
        "pred": "",
    }
    for name, value in subrewards.items():
        info[name] = float(value)
    return info


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """PD-GDPO ``compute_score`` entrypoint.

    Returns a dict where ``"score"`` is the primary reward and every other
    key is either a heuristic sub-reward (consumed by the advantage
    estimator) or a logging field (ignored by the estimator).
    """
    # Allow callers to opt-in to component emission per-category. By
    # default we emit components only when explicitly enabled; downstream
    # configs typically set ``math_enable_sub_rewards=true`` etc.
    if _is_math(data_source):
        return _score_math(data_source, solution_str, ground_truth, extra_info, kwargs)

    if _is_coding(data_source):
        # Coding tasks always go through _score_coding so the primary
        # reward gets computed (e.g. via SandboxFusion for deepcoder).
        # collect_subrewards inside it respects coding_enable_sub_rewards,
        # so GRPO baselines pass coding_enable_sub_rewards=false and
        # still get the primary signal.
        return _score_coding(data_source, solution_str, ground_truth, extra_info, kwargs)

    # Fallback: pass-through the verl default scorer.
    base_res = default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
    if isinstance(base_res, dict):
        if "score" not in base_res and "acc" in base_res:
            base_res = {**base_res, "score": float(base_res["acc"])}
        return base_res
    return base_res
