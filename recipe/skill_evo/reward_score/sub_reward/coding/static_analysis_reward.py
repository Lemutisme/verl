import ast
from typing import Any

from ..common import extract_code, parse_python


def compute(ctx: dict[str, Any], **_: Any) -> float:
    code = ctx.get("code") or extract_code(ctx.get("response", ""))
    tree = parse_python(code)
    if tree is None:
        return 0.0

    score = 1.0
    has_callable = False
    top_returns = 0
    risky_calls = {"eval", "exec", "compile", "__import__"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            has_callable = True
        if isinstance(node, ast.Return):
            parent = getattr(node, "_parent", None)
            if isinstance(parent, ast.Module):
                top_returns += 1
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in risky_calls:
                score -= 0.20
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.name.split(".")[0] for alias in node.names}
            if names & {"os", "subprocess", "socket", "requests", "shutil"}:
                score -= 0.10

    if not has_callable:
        score -= 0.10
    if top_returns:
        score -= 0.10
    return score
