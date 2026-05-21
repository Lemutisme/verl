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

from reward_score.sub_reward import collect_subrewards, to_bool, clip
import reward_score.coding_executable_reward as coding_evaluator

# Register local advantage estimators when this module is imported
try:
    import pdpo_init  # noqa: F401
except ImportError:
    pass


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


_LABELED_ANSWER_RE = re.compile(r"(?i)(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)")
_LATEX_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_combine_mode(value: Any) -> str:
    mode = str(value or "none").strip().lower().replace("_", "-")
    aliases = {
        "": "none",
        "none": "none",
        "original": "none",
        "ori": "none",
        "new": "multiplier",
        "new-reward": "multiplier",
        "static": "multiplier",
        "multiplier": "multiplier",
        "pdpo": "pdpo",
        "pdpo-reward": "pdpo",
        "gdpo": "gdpo",
        "gdpo-reward": "gdpo",
    }
    if mode in aliases:
        return aliases[mode]
    raise ValueError(f"Unsupported combine_mode: {value!r}; expected none, multiplier/new, pdpo, or gdpo")


def _is_advantage_aux_mode(combine_mode: str) -> bool:
    return combine_mode in {"pdpo", "gdpo"}


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


def _pdpo_reward_info(
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


def _coding_pdpo_reward_info(
    main_reward: float,
    subrewards: dict[str, float],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    partial_pass_rate = max(0.0, min(1.0, float(main_reward)))
    strict_threshold = _to_float(kwargs.get("coding_strict_acc_threshold"), 1.0)
    return _pdpo_reward_info(
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


def _static_reward_info(
    main_reward: float,
    subrewards: dict[str, float],
    kwargs: dict[str, Any],
    *,
    signed: bool = False,
    acc: bool | None = None,
    original_reward: float | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reward = float(main_reward)
    for name, value in subrewards.items():
        reward += _subreward_weight(name, kwargs) * float(value)
    reward = clip(reward, 0.0, 1.0)
    score = 2.0 * reward - 1.0 if signed else reward

    info = {
        "score": float(score),
        "combined_reward": float(score),
        "main_reward": float(main_reward),
        "acc": bool(main_reward > 0.0) if acc is None else bool(acc),
        "original_reward": float(main_reward if original_reward is None else original_reward),
    }
    if extra_metrics:
        info.update(extra_metrics)
    for name, value in subrewards.items():
        info[f"{name}_reward"] = float(value)
        info[f"weight_{name}"] = float(_subreward_weight(name, kwargs))
    return _flatten_subrewards(info, subrewards)


def _coding_main_reward_info(main_reward: float, kwargs: dict[str, Any]) -> dict[str, Any]:
    partial_pass_rate = max(0.0, min(1.0, float(main_reward)))
    strict_threshold = _to_float(kwargs.get("coding_strict_acc_threshold"), 1.0)
    return {
        "score": partial_pass_rate,
        "combined_reward": partial_pass_rate,
        "main_reward": partial_pass_rate,
        "acc": partial_pass_rate >= strict_threshold,
        "original_reward": partial_pass_rate,
        "partial_pass_rate": partial_pass_rate,
        "any_pass": partial_pass_rate > 0.0,
    }


def _coding_static_reward_info(
    main_reward: float,
    subrewards: dict[str, float],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    partial_pass_rate = max(0.0, min(1.0, float(main_reward)))
    strict_threshold = _to_float(kwargs.get("coding_strict_acc_threshold"), 1.0)
    return _static_reward_info(
        partial_pass_rate,
        subrewards,
        kwargs,
        signed=False,
        acc=partial_pass_rate >= strict_threshold,
        original_reward=partial_pass_rate,
        extra_metrics={
            "partial_pass_rate": partial_pass_rate,
            "any_pass": partial_pass_rate > 0.0,
        },
    )


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

    combine_mode = _normalize_combine_mode(kwargs.get("combine_mode", "none"))
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

    # Advantage-level process modes return separate channels and skip reward-level combination.
    if _is_advantage_aux_mode(combine_mode):
        info = _pdpo_reward_info(
            1.0 if base_acc else 0.0,
            subrewards,
            kwargs,
            signed=to_bool(kwargs.get("math_signed_reward", True), True),
            acc=base_acc,
        )
        info["base_math_score"] = float(base_score)
        info["original_reward"] = float(base_score)
        return info

    main_reward = 1.0 if base_acc else 0.0
    info = _static_reward_info(
        main_reward,
        subrewards,
        kwargs,
        signed=to_bool(kwargs.get("math_signed_reward", True), True),
        acc=base_acc,
        original_reward=float(base_score),
    )
    info["base_math_score"] = float(base_score)
    if isinstance(base_res, dict) and "pred" in base_res:
        info["pred"] = base_res["pred"]
    return info

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    ds_str = str(data_source)
    if ds_str in MATH_DATA_SOURCES or ds_str.startswith("aime") or ds_str.startswith("deepscalar") or "math" in ds_str:
        return _score_math(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)

    combine_mode = _normalize_combine_mode(kwargs.get("combine_mode", "none"))
        
    if data_source in ["mbpp:train", "mbpp:test", "mbpp:validation", "mbpp"]:
        # Pass return_components=True to just get the raw components
        res = coding_evaluator.compute_score_coding(
            solution_str,
            ground_truth,
            eval_mode="assert",
            return_components=True,
            **kwargs,
        )
        
        # Determine if it's a batch or single
        if not isinstance(res, (list, dict)):
            # Fallback: tuple (main_reward, subrewards)
            main_reward, subrewards = res
            res = {"main_reward": main_reward, "subrewards": subrewards}

        if _is_advantage_aux_mode(combine_mode):
            if isinstance(res, list):
                return [
                    _coding_pdpo_reward_info(r["main_reward"], r["subrewards"], kwargs)
                    for r in res
                ]
            return _coding_pdpo_reward_info(res["main_reward"], res["subrewards"], kwargs)

        if isinstance(res, list):
            if combine_mode == "none":
                infos = [_coding_main_reward_info(r["main_reward"], kwargs) for r in res]
            else:
                infos = [_coding_static_reward_info(r["main_reward"], r["subrewards"], kwargs) for r in res]
            return [info["score"] for info in infos]

        if combine_mode == "none":
            return _coding_main_reward_info(res["main_reward"], kwargs)
        return _coding_static_reward_info(res["main_reward"], res["subrewards"], kwargs)
            
    elif _is_coding_data_source(data_source):
        sandbox_fusion_url = kwargs.get("sandbox_fusion_url", None)
        concurrent_semaphore = kwargs.get("concurrent_semaphore", None)

        res = coding_evaluator.compute_score_coding(
            solution_str, ground_truth, 
            eval_mode="stdio",
            sandbox_url=sandbox_fusion_url, 
            concurrent_semaphore=concurrent_semaphore, 
            return_components=True, 
            **kwargs
        )
        
        if not isinstance(res, (list, dict)):
            # Fallback: tuple (main_reward, subrewards)
            main_reward, subrewards = res
            res = {"main_reward": main_reward, "subrewards": subrewards}

        if _is_advantage_aux_mode(combine_mode):
            if isinstance(res, list):
                return [
                    _coding_pdpo_reward_info(r["main_reward"], r["subrewards"], kwargs)
                    for r in res
                ]
            return _coding_pdpo_reward_info(res["main_reward"], res["subrewards"], kwargs)

        if isinstance(res, list):
            if combine_mode == "none":
                infos = [_coding_main_reward_info(r["main_reward"], kwargs) for r in res]
            else:
                infos = [_coding_static_reward_info(r["main_reward"], r["subrewards"], kwargs) for r in res]
            return [info["score"] for info in infos]

        if combine_mode == "none":
            return _coding_main_reward_info(res["main_reward"], kwargs)
        return _coding_static_reward_info(res["main_reward"], res["subrewards"], kwargs)

    # Fallback to the default compute score for other data sources
    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
