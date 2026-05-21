import ast
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ..common import response_text, to_float
from .efficiency_utils import approx_token_count, response_parts


GSM8K_UNIT_RE = re.compile(r"<<\s*([^<>=]+?)\s*=\s*([^<>]+?)\s*>>")
FREE_EQUATION_RE = re.compile(
    r"(?<![\w<])"
    r"([-+]?\$?\d[\d,.\s+\-*/^()×÷$]*?[+\-*/^×÷]\s*[-+]?\$?\d[\d,.\s+\-*/^()×÷$]*?)"
    r"\s*=\s*"
    r"([-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?)"
)
ANSWER_CANDIDATE_RES = [
    re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?)"),
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"(?i)(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)"),
]


class UnsafeExpressionError(ValueError):
    pass


@dataclass(frozen=True)
class ArithmeticClaim:
    expression: str
    expected_text: str
    raw: str
    source: str
    start: int
    actual_value: Fraction | None
    expected_value: Fraction | None
    parseable: bool
    valid: bool
    error: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (_canonical_expression(self.expression), _canonical_expression(self.expected_text))


@dataclass(frozen=True)
class VerificationReport:
    claims: tuple[ArithmeticClaim, ...]
    gold_units: tuple[ArithmeticClaim, ...]
    non_final_gold_units: tuple[ArithmeticClaim, ...]
    matched_non_final_gold_units: int
    valid_claim_count: int
    unique_valid_claim_count: int
    duplicate_ratio: float
    prefix_validity_score: float
    post_answer_tokens: int
    final_answer_extractable: bool


def _clean_expression(expr: Any) -> str:
    text = str(expr or "").strip()
    text = text.replace("$", "")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("×", "*").replace("÷", "/")
    text = text.replace("^", "**")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return text


def _canonical_expression(expr: Any) -> str:
    return re.sub(r"\s+", "", _clean_expression(expr))


def _eval_ast(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(f"unsupported constant: {node.value!r}")
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp):
        value = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise UnsafeExpressionError(f"unsupported unary operator: {ast.dump(node.op)}")
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise UnsafeExpressionError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1:
                raise UnsafeExpressionError("fractional exponents are unsupported")
            exponent = right.numerator
            if abs(exponent) > 12:
                raise UnsafeExpressionError("exponent too large")
            return left**exponent
        raise UnsafeExpressionError(f"unsupported binary operator: {ast.dump(node.op)}")
    raise UnsafeExpressionError(f"unsupported node: {ast.dump(node)}")


def safe_eval_fraction(expr: Any, *, max_chars: int = 200) -> Fraction:
    text = _clean_expression(expr)
    if not text:
        raise UnsafeExpressionError("empty expression")
    if len(text) > max_chars:
        raise UnsafeExpressionError("expression too long")
    if not re.fullmatch(r"[0-9\s+\-*/().]+", text):
        raise UnsafeExpressionError("unsupported characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError("invalid expression syntax") from exc
    return _eval_ast(tree)


def _close(a: Fraction | None, b: Fraction | None, tol: float) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a - b)) <= tol


def _make_claim(expr: str, expected: str, raw: str, source: str, start: int, kwargs: dict[str, Any]) -> ArithmeticClaim:
    max_chars = int(float(kwargs.get("math_executable_max_expr_chars", 200)))
    tol = to_float(kwargs.get("math_executable_numeric_tol"), 1e-6)
    try:
        actual = safe_eval_fraction(expr, max_chars=max_chars)
        expected_value = safe_eval_fraction(expected, max_chars=max_chars)
        valid = _close(actual, expected_value, tol)
        return ArithmeticClaim(
            expression=expr,
            expected_text=expected,
            raw=raw,
            source=source,
            start=start,
            actual_value=actual,
            expected_value=expected_value,
            parseable=True,
            valid=valid,
        )
    except UnsafeExpressionError as exc:
        return ArithmeticClaim(
            expression=expr,
            expected_text=expected,
            raw=raw,
            source=source,
            start=start,
            actual_value=None,
            expected_value=None,
            parseable=False,
            valid=False,
            error=str(exc),
        )


def _extract_gsm8k_units(text: str, source: str, kwargs: dict[str, Any]) -> list[ArithmeticClaim]:
    claims = []
    for match in GSM8K_UNIT_RE.finditer(text or ""):
        raw = match.group(0)
        claims.append(_make_claim(match.group(1), match.group(2), raw, source, match.start(), kwargs))
    return claims


def _blank_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for idx in range(start, min(end, len(chars))):
            chars[idx] = " "
    return "".join(chars)


def _extract_free_equations(text: str, kwargs: dict[str, Any]) -> list[ArithmeticClaim]:
    spans = [match.span() for match in GSM8K_UNIT_RE.finditer(text or "")]
    search_text = _blank_spans(text or "", spans)
    claims = []
    for match in FREE_EQUATION_RE.finditer(search_text):
        raw = match.group(0)
        claims.append(_make_claim(match.group(1), match.group(2), raw, "free_text", match.start(), kwargs))
    return claims


def _extract_generated_claims(text: str, kwargs: dict[str, Any]) -> tuple[ArithmeticClaim, ...]:
    max_claims = int(float(kwargs.get("math_executable_max_claims", 32)))
    claims = _extract_gsm8k_units(text, "gsm8k_inline", kwargs)
    claims.extend(_extract_free_equations(text, kwargs))
    claims.sort(key=lambda claim: claim.start)
    return tuple(claims[:max(0, max_claims)])


def _ground_truth_value(ctx: dict[str, Any], kwargs: dict[str, Any]) -> Fraction | None:
    try:
        max_chars = int(float(kwargs.get("math_executable_max_expr_chars", 200)))
        return safe_eval_fraction(ctx.get("ground_truth"), max_chars=max_chars)
    except UnsafeExpressionError:
        return None


def _extract_final_answer(text: str, kwargs: dict[str, Any]) -> bool:
    max_chars = int(float(kwargs.get("math_executable_max_expr_chars", 200)))
    for regex in ANSWER_CANDIDATE_RES:
        for match in regex.finditer(text or ""):
            candidate = str(match.group(1)).strip().split()[0]
            try:
                safe_eval_fraction(candidate, max_chars=max_chars)
                return True
            except UnsafeExpressionError:
                continue
    return False


def _non_final_gold_units(ctx: dict[str, Any], gold_units: tuple[ArithmeticClaim, ...], kwargs: dict[str, Any]) -> tuple[ArithmeticClaim, ...]:
    tol = to_float(kwargs.get("math_executable_numeric_tol"), 1e-6)
    gt_value = _ground_truth_value(ctx, kwargs)
    non_final = []
    for claim in gold_units:
        if gt_value is not None and _close(claim.expected_value, gt_value, tol):
            continue
        non_final.append(claim)
    return tuple(non_final)


def _claims_match(generated: ArithmeticClaim, gold: ArithmeticClaim, tol: float) -> bool:
    if not generated.valid or not gold.valid:
        return False
    if generated.key == gold.key:
        return True
    return _close(generated.expected_value, gold.expected_value, tol)


def _prefix_validity_score(claims: tuple[ArithmeticClaim, ...]) -> float:
    if not claims:
        return 0.0
    prefix_ok = True
    checkpoints = []
    for claim in claims:
        prefix_ok = prefix_ok and claim.valid
        checkpoints.append(1.0 if prefix_ok else 0.0)
    return sum(checkpoints) / float(len(checkpoints))


def _duplicate_ratio(valid_claims: list[ArithmeticClaim]) -> tuple[int, float]:
    if not valid_claims:
        return 0, 0.0
    keys = [claim.key for claim in valid_claims]
    unique_count = len(set(keys))
    duplicate_count = len(keys) - unique_count
    return unique_count, duplicate_count / float(len(keys))


def get_verification_report(ctx: dict[str, Any], **kwargs: Any) -> VerificationReport:
    tol = to_float(kwargs.get("math_executable_numeric_tol"), 1e-6)
    max_claims = int(float(kwargs.get("math_executable_max_claims", 32)))
    cache_key = (tol, max_claims)
    cache = ctx.setdefault("_math_executable_report_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    response = response_text(ctx)
    extra_info = ctx.get("extra_info") if isinstance(ctx.get("extra_info"), dict) else {}
    gold_text = str(extra_info.get("answer") or "")

    claims = _extract_generated_claims(response, kwargs)
    gold_units = tuple(claim for claim in _extract_gsm8k_units(gold_text, "gsm8k_gold", kwargs) if claim.valid)
    non_final_gold_units = _non_final_gold_units(ctx, gold_units, kwargs)
    matched = sum(1 for gold in non_final_gold_units if any(_claims_match(claim, gold, tol) for claim in claims))

    valid_claims = [claim for claim in claims if claim.valid]
    unique_valid_count, duplicate_ratio = _duplicate_ratio(valid_claims)
    _, _, answer_suffix = response_parts(ctx)

    report = VerificationReport(
        claims=claims,
        gold_units=gold_units,
        non_final_gold_units=non_final_gold_units,
        matched_non_final_gold_units=matched,
        valid_claim_count=len(valid_claims),
        unique_valid_claim_count=unique_valid_count,
        duplicate_ratio=duplicate_ratio,
        prefix_validity_score=_prefix_validity_score(claims),
        post_answer_tokens=approx_token_count(answer_suffix),
        final_answer_extractable=_extract_final_answer(response, kwargs),
    )
    cache[cache_key] = report
    return report


def cap_score_if_wrong(ctx: dict[str, Any], score: float, kwargs: dict[str, Any]) -> float:
    score = max(0.0, min(1.0, float(score)))
    if bool(ctx.get("base_acc", False)):
        return score
    wrong_cap = to_float(kwargs.get("math_executable_wrong_cap"), 0.35)
    return min(score, max(0.0, min(1.0, wrong_cap)))
