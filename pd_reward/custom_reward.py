import os
import re
import sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

# Add current directory to sys.path to allow importing local reward_score package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_dapo

from reward_score.primal_dual_core import GenericRewardCombiner
from reward_score.sub_reward import collect_subrewards, weight_overrides, to_bool, clip
import reward_score.mbpp_action_thought_reward as mbpp_evaluator
import reward_score.deepcoder_action_thought_reward as deepcoder_evaluator

# Register PDAR advantage estimator when this module is imported
try:
    import pdar_init  # noqa: F401 — triggers @register_adv_est("pdar")
except ImportError:
    pass  # PDAR not available in this installation

# Initialize the generic combiner (will parse kwargs for combine_mode="pd"|"multiplier")
# We delay initialization until the first call to ensure we have the kwargs


MATH_DATA_SOURCES = {
    "general365",
    "openr1_math_220k",
    "deepscalar",
    "math_dapo",
    "math",
    "math_dapo_reasoning",
    "openai/gsm8k",
    "gsm8k",
    "amc",
    "aops",
    "cn_k12",
    "math500",
    "numina",
    "synthetic",
    "olympiad",
}


CODING_DATA_SOURCES = {
    "apps",
    "code_contests",
    "codecontests",
    "codeforces",
    "taco",
}


_COMBINER_CACHE = {}


_LABELED_ANSWER_RE = re.compile(r"(?i)(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)")
_LATEX_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extra_info_dict(extra_info: Any) -> dict[str, Any]:
    return extra_info if isinstance(extra_info, dict) else {}


def _global_step(extra_info: Any) -> int:
    extra = _extra_info_dict(extra_info)
    for key in ("global_step", "global_steps"):
        if key in extra and extra[key] is not None:
            try:
                return int(extra[key])
            except (TypeError, ValueError):
                return -1
    return -1


def _should_update_dual(extra_info: Any) -> bool:
    return not to_bool(_extra_info_dict(extra_info).get("is_validation", False), False)


def _subreward_weight(name: str, kwargs: dict[str, Any]) -> float:
    candidates = [f"weight_{name}"]
    for category in ("math", "coding"):
        prefix = f"{category}_"
        if name.startswith(prefix):
            candidates.append(f"{category}_weight_{name[len(prefix):]}")
    for key in candidates:
        if key in kwargs:
            return _to_float(kwargs[key], 1.0)
    return 1.0


def _weighted_subreward_average(subrewards: dict[str, float], kwargs: dict[str, Any]) -> float:
    if not subrewards:
        return 0.0
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, value in subrewards.items():
        weight = _subreward_weight(name, kwargs)
        if weight <= 0:
            continue
        weighted_sum += weight * float(value)
        weight_sum += weight
    if weight_sum <= 0:
        return sum(float(v) for v in subrewards.values()) / max(len(subrewards), 1)
    return weighted_sum / weight_sum


def _extract_boxed_answers(text: str) -> list[str]:
    answers = []
    for match in re.finditer(r"\\boxed\s*\{", str(text or "")):
        start = match.end()
        depth = 1
        idx = start
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            answers.append(text[start : idx - 1].strip())
    return answers


def _clean_candidate_answer(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s+", "", text)
    text = re.sub(r"(?i)^(?:the\s+)?(?:final\s+answer|answer)\s*[:：]\s*", "", text).strip()
    text = text.strip().strip("$").strip()
    boxed_answers = _extract_boxed_answers(text)
    if len(boxed_answers) == 1:
        text = boxed_answers[0]
    return text.strip().strip("$").strip().strip(".。")


def _normalize_math_answer(value: Any) -> str:
    return math_dapo.normalize_final_answer(_clean_candidate_answer(value))


def _fraction_from_answer(value: Any) -> Fraction | None:
    text = _normalize_math_answer(value).replace(" ", "")
    if not text:
        return None

    sign = 1
    if text.startswith("-"):
        sign = -1
        text = text[1:]
    elif text.startswith("+"):
        text = text[1:]

    frac_match = _LATEX_FRAC_RE.fullmatch(text)
    if frac_match:
        try:
            numerator = Decimal(frac_match.group(1))
            denominator = Decimal(frac_match.group(2))
            if denominator == 0:
                return None
            return sign * Fraction(numerator) / Fraction(denominator)
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    slash_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", text)
    if slash_match:
        try:
            denominator = Decimal(slash_match.group(2))
            if denominator == 0:
                return None
            return sign * Fraction(Decimal(slash_match.group(1))) / Fraction(denominator)
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            return sign * Fraction(Decimal(text))
        except (InvalidOperation, ValueError):
            return None
    return None


def _math_answers_equivalent(pred: Any, ground_truth: Any) -> tuple[bool, str]:
    pred_norm = _normalize_math_answer(pred)
    gt_norm = _normalize_math_answer(ground_truth)
    if pred_norm == gt_norm:
        return True, pred_norm

    pred_fraction = _fraction_from_answer(pred_norm)
    gt_fraction = _fraction_from_answer(gt_norm)
    if pred_fraction is not None and gt_fraction is not None and pred_fraction == gt_fraction:
        return True, pred_norm

    return False, pred_norm


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def _math_answer_candidates(solution_str: str) -> list[str]:
    candidates: list[str] = []
    text = str(solution_str or "")

    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        if tail:
            candidates.append(_first_nonempty_line(tail))

    labeled_matches = list(_LABELED_ANSWER_RE.finditer(text))
    if labeled_matches:
        candidates.append(labeled_matches[-1].group(1))

    boxed_answers = _extract_boxed_answers(text)
    if boxed_answers:
        candidates.append(boxed_answers[-1])

    seen = set()
    deduped = []
    for candidate in candidates:
        cleaned = _clean_candidate_answer(candidate)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def _is_coding_data_source(data_source: Any) -> bool:
    source = str(data_source or "").strip().lower().replace("-", "_")
    return source.startswith("deepcoder") or source.startswith("eurus") or source in CODING_DATA_SOURCES


def _score_math_dapo_flexible(solution_str: str, ground_truth: str):
    base_res = math_dapo.compute_score(solution_str, ground_truth)
    if isinstance(base_res, dict) and bool(base_res.get("acc", False)):
        return base_res

    for candidate in _math_answer_candidates(solution_str):
        correct, pred = _math_answers_equivalent(candidate, ground_truth)
        if correct:
            return {"score": 1.0, "acc": True, "pred": pred}
    return base_res


def _flatten_subrewards(info: dict[str, Any], subrewards: dict[str, float]) -> dict[str, Any]:
    for name, value in subrewards.items():
        info[name] = float(value)
    return info


def _pdar_reward_info(
    main_reward: float,
    subrewards: dict[str, float],
    kwargs: dict[str, Any],
    *,
    signed: bool = False,
    acc: bool | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    main = float(main_reward)
    if signed:
        main = 1.0 if main > 0.0 else -1.0
    info = {
        "score": main,
        "main_reward": main,
        "aux_reward_combined": _weighted_subreward_average(subrewards, kwargs),
        "aux_rewards": subrewards,
        "acc": bool(main_reward > 0.0) if acc is None else bool(acc),
        "original_reward": float(main_reward),
    }
    if extra_metrics:
        info.update(extra_metrics)
    return _flatten_subrewards(info, subrewards)


def _coding_pdar_reward_info(
    main_reward: float,
    subrewards: dict[str, float],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    partial_pass_rate = max(0.0, min(1.0, float(main_reward)))
    strict_threshold = _to_float(kwargs.get("coding_strict_acc_threshold"), 1.0)
    return _pdar_reward_info(
        main_reward,
        subrewards,
        kwargs,
        signed=False,
        acc=partial_pass_rate >= strict_threshold,
        extra_metrics={
            "partial_pass_rate": partial_pass_rate,
            "any_pass": partial_pass_rate > 0.0,
        },
    )


def _get_combiner(kwargs: dict[str, Any]) -> GenericRewardCombiner:

    combine_mode = str(kwargs.get("combine_mode", "none")).lower()
    if combine_mode not in _COMBINER_CACHE:
        combiner_kwargs = {k: v for k, v in kwargs.items() if k not in ["combine_mode"]}
        combiner_kwargs.update(weight_overrides("coding", **kwargs))
        combiner_kwargs.update(weight_overrides("math", **kwargs))
        _COMBINER_CACHE[combine_mode] = GenericRewardCombiner(combine_mode=combine_mode, subreward_names=[], **combiner_kwargs)
    
    return _COMBINER_CACHE[combine_mode]


def _score_math(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in {"openai/gsm8k", "gsm8k"}:
        from verl.utils.reward_score import gsm8k

        base_res = gsm8k.compute_score(solution_str, ground_truth)
        
        # Fallback for models that output \boxed{} instead of ####
        base_wrong = (
            (isinstance(base_res, (int, float)) and base_res == 0.0)
            or (isinstance(base_res, dict) and base_res.get("score", 0.0) == 0.0)
        )
        if base_wrong:
            base_res = gsm8k.compute_score(solution_str, ground_truth, method="flexible")
            flexible_wrong = (
                (isinstance(base_res, (int, float)) and base_res == 0.0)
                or (isinstance(base_res, dict) and not bool(base_res.get("acc", base_res.get("score", 0.0) > 0.0)))
            )
            if flexible_wrong:
                base_res = _score_math_dapo_flexible(solution_str, ground_truth)
                
        if isinstance(base_res, dict):
            base_score = float(base_res.get("score", base_res.get("acc", 0.0)))
            base_acc = bool(base_res.get("acc", base_score > 0.0))
        else:
            base_score = float(base_res)
            base_acc = base_score > 0.0
            base_res = {"score": base_score, "acc": base_acc}
    else:
        base_res = _score_math_dapo_flexible(solution_str, ground_truth)
        base_score = float(base_res.get("score", 0.0)) if isinstance(base_res, dict) else float(base_res)
        base_acc = bool(base_res.get("acc", base_score > 0.0)) if isinstance(base_res, dict) else base_score > 0.0

    combine_mode = str(kwargs.get("combine_mode", "none")).lower()
    if combine_mode == "none" or not to_bool(kwargs.get("math_enable_sub_rewards", False), False):
        if to_bool(kwargs.get("math_signed_reward", True), True):
            signed_score = 1.0 if base_acc else -1.0
            if isinstance(base_res, dict):
                base_res["score"] = signed_score
            else:
                base_res = signed_score
        return base_res

    ctx = {
        "response": solution_str,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info or {},
        "data_source": data_source,
        "base_score": base_score,
        "base_acc": base_acc,
    }
    subrewards = collect_subrewards("math", ctx, **kwargs)
    if not subrewards:
        if to_bool(kwargs.get("math_signed_reward", True), True):
            signed_score = 1.0 if base_acc else -1.0
            if isinstance(base_res, dict):
                base_res["score"] = signed_score
            else:
                base_res = signed_score
        return base_res

    # --- PDAR mode: return separate channels, skip reward-level combination ---
    if combine_mode == "pdar":
        info = _pdar_reward_info(
            1.0 if base_acc else 0.0,
            subrewards,
            kwargs,
            signed=to_bool(kwargs.get("math_signed_reward", True), True),
            acc=base_acc,
        )
        info["base_math_score"] = float(base_score)
        info["original_reward"] = float(base_score)
        return info

    combiner = _get_combiner(kwargs)
    main_reward = 1.0 if base_acc else 0.0
    info = combiner.process_batch(
        [main_reward],
        [subrewards],
        global_step=_global_step(extra_info),
        update_dual=_should_update_dual(extra_info),
    )[0]

    if to_bool(kwargs.get("math_signed_reward", True), True):
        if combine_mode == "pd":
            # PD rewards are already in [-1, 1] from process_batch clipping;
            # remap to signed scale: positive for correct, negative for wrong
            info["score"] = clip(info["score"], -1.0, 1.0)
        else:
            info["score"] = 2.0 * clip(info["score"], 0.0, 1.0) - 1.0
        info["combined_reward"] = info["score"]

    info["base_math_score"] = float(base_score)
    info["original_reward"] = float(base_score)
    info["acc"] = bool(base_acc)
    _flatten_subrewards(info, subrewards)
    if isinstance(base_res, dict) and "pred" in base_res:
        info["pred"] = base_res["pred"]
    return info

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    ds_str = str(data_source)
    if ds_str in MATH_DATA_SOURCES or ds_str.startswith("aime") or ds_str.startswith("deepscalar") or "math" in ds_str:
        return _score_math(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)

    combine_mode = str(kwargs.get("combine_mode", "none")).lower()
    combiner = _get_combiner(kwargs)
        
    if data_source in ["mbpp:train", "mbpp:test", "mbpp:validation", "mbpp"]:
        # Pass return_components=True to just get the raw components
        res = mbpp_evaluator.compute_score_mbpp(solution_str, ground_truth, return_components=True, **kwargs)
        
        # Determine if it's a batch or single
        if not isinstance(res, (list, dict)):
            # Fallback: tuple (main_reward, subrewards)
            main_reward, subrewards = res
            res = {"main_reward": main_reward, "subrewards": subrewards}

        if combine_mode == "pdar":
            if isinstance(res, list):
                return [
                    _coding_pdar_reward_info(r["main_reward"], r["subrewards"], kwargs)
                    for r in res
                ]
            return _coding_pdar_reward_info(res["main_reward"], res["subrewards"], kwargs)

        if isinstance(res, list):
            main_rewards = [r["main_reward"] for r in res]
            subrewards_list = [r["subrewards"] for r in res]
            infos = combiner.process_batch(
                main_rewards,
                subrewards_list,
                global_step=_global_step(extra_info),
                update_dual=_should_update_dual(extra_info),
            )
            return [info["score"] for info in infos]
        else:
            infos = combiner.process_batch(
                [res["main_reward"]],
                [res["subrewards"]],
                global_step=_global_step(extra_info),
                update_dual=_should_update_dual(extra_info),
            )
            return infos[0] if infos else 0.0
            
    elif _is_coding_data_source(data_source):
        sandbox_fusion_url = kwargs.get("sandbox_fusion_url", None)
        concurrent_semaphore = kwargs.get("concurrent_semaphore", None)

        res = deepcoder_evaluator.compute_score_deepcoder(
            solution_str, ground_truth, 
            sandbox_url=sandbox_fusion_url, 
            concurrent_semaphore=concurrent_semaphore, 
            return_components=True, 
            **kwargs
        )
        
        if not isinstance(res, (list, dict)):
            # Fallback: tuple (main_reward, subrewards)
            main_reward, subrewards = res
            res = {"main_reward": main_reward, "subrewards": subrewards}

        if combine_mode == "pdar":
            if isinstance(res, list):
                return [
                    _coding_pdar_reward_info(r["main_reward"], r["subrewards"], kwargs)
                    for r in res
                ]
            return _coding_pdar_reward_info(res["main_reward"], res["subrewards"], kwargs)

        if isinstance(res, list):
            main_rewards = [r["main_reward"] for r in res]
            subrewards_list = [r["subrewards"] for r in res]
            infos = combiner.process_batch(
                main_rewards,
                subrewards_list,
                global_step=_global_step(extra_info),
                update_dual=_should_update_dual(extra_info),
            )
            return [info["score"] for info in infos]
        else:
            infos = combiner.process_batch(
                [res["main_reward"]],
                [res["subrewards"]],
                global_step=_global_step(extra_info),
                update_dual=_should_update_dual(extra_info),
            )
            return infos[0] if infos else 0.0

    # Fallback to the default compute score for other data sources
    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
