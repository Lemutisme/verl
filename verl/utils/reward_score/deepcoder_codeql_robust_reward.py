import math
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CodeQLRobustnessResult:
    ok: bool
    error: str
    scalar: float
    num_nodes: int
    num_edges: int
    density: float


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalize_codeql_scalar(scalar: float, threshold: float = 0.82, scale: float = 0.15) -> float:
    scale = max(1e-6, float(scale))
    return _clip01((float(scalar) - float(threshold)) / scale)


def _run_cmd(cmd: List[str], timeout_s: int) -> Dict[str, object]:
    rec: Dict[str, object] = {"ok": False, "returncode": None, "error": ""}
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        rec["error"] = f"TIMEOUT({timeout_s}s)"
        return rec
    except Exception as exc:
        rec["error"] = f"EXEC_ERROR: {exc}"
        return rec

    rec["returncode"] = proc.returncode
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if len(err) > 500:
            err = err[:500] + "..."
        rec["error"] = f"RC={proc.returncode}: {err}"
        return rec
    rec["ok"] = True
    return rec


def _write_queries(query_dir: Path) -> Dict[str, Path]:
    query_dir.mkdir(parents=True, exist_ok=True)
    qlpack_path = query_dir / "qlpack.yml"
    edges_query = query_dir / "local_flow_edges.ql"
    nodes_query = query_dir / "dataflow_nodes.ql"

    qlpack_path.write_text(
        textwrap.dedent(
            """
            name: local/deepcoder-reward-dataflow
            version: 0.0.0
            dependencies:
              codeql/python-all: "*"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    edges_query.write_text(
        textwrap.dedent(
            """
            import python
            import semmle.python.dataflow.new.DataFlow

            from DataFlow::Node src, DataFlow::Node dst
            where DataFlow::localFlow(src, dst)
            select src, dst
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    nodes_query.write_text(
        textwrap.dedent(
            """
            import python
            import semmle.python.dataflow.new.DataFlow

            from DataFlow::Node n
            select n
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    return {"qlpack": qlpack_path, "edges_query": edges_query, "nodes_query": nodes_query}


def _count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)


def compute_codeql_robustness(
    code: str,
    codeql_bin: str = "",
    timeout_s: int = 120,
    workdir: Optional[str] = None,
) -> CodeQLRobustnessResult:
    if not isinstance(code, str) or not code.strip():
        return CodeQLRobustnessResult(False, "EMPTY_CODE", 0.0, 0, 0, 0.0)

    codeql_path = codeql_bin.strip() if isinstance(codeql_bin, str) else ""
    if not codeql_path:
        found = shutil.which("codeql")
        if not found:
            return CodeQLRobustnessResult(False, "CODEQL_NOT_FOUND", 0.0, 0, 0, 0.0)
        codeql_path = found

    timeout_s = max(1, int(timeout_s))
    tmp_parent = None
    if isinstance(workdir, str):
        stripped = workdir.strip()
        if stripped:
            tmp_parent = stripped

    try:
        with tempfile.TemporaryDirectory(prefix="deepcoder_reward_codeql_", dir=tmp_parent) as tmp:
            tmp_dir = Path(tmp)
            src_dir = tmp_dir / "src"
            db_dir = tmp_dir / "db"
            query_dir = tmp_dir / "queries"
            out_dir = tmp_dir / "out"
            src_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            code_path = src_dir / "snippet.py"
            code_path.write_text(code + "\n", encoding="utf-8")

            queries = _write_queries(query_dir)
            edges_query = queries["edges_query"]
            nodes_query = queries["nodes_query"]

            rec = _run_cmd(
                [
                    codeql_path,
                    "database",
                    "create",
                    str(db_dir),
                    "--language=python",
                    "--source-root",
                    str(src_dir),
                    "--overwrite",
                ],
                timeout_s=timeout_s,
            )
            if not bool(rec["ok"]):
                return CodeQLRobustnessResult(False, f"DB_CREATE_FAILED: {rec['error']}", 0.0, 0, 0, 0.0)

            edge_bqrs = out_dir / "edges.bqrs"
            edge_csv = out_dir / "edges.csv"
            node_bqrs = out_dir / "nodes.bqrs"
            node_csv = out_dir / "nodes.csv"

            for cmd, err_tag in [
                (
                    [
                        codeql_path,
                        "query",
                        "run",
                        str(edges_query),
                        "--database",
                        str(db_dir),
                        "--output",
                        str(edge_bqrs),
                    ],
                    "QUERY_EDGES_FAILED",
                ),
                (
                    [
                        codeql_path,
                        "bqrs",
                        "decode",
                        str(edge_bqrs),
                        "--format=csv",
                        "--output",
                        str(edge_csv),
                    ],
                    "DECODE_EDGES_FAILED",
                ),
                (
                    [
                        codeql_path,
                        "query",
                        "run",
                        str(nodes_query),
                        "--database",
                        str(db_dir),
                        "--output",
                        str(node_bqrs),
                    ],
                    "QUERY_NODES_FAILED",
                ),
                (
                    [
                        codeql_path,
                        "bqrs",
                        "decode",
                        str(node_bqrs),
                        "--format=csv",
                        "--output",
                        str(node_csv),
                    ],
                    "DECODE_NODES_FAILED",
                ),
            ]:
                rec = _run_cmd(cmd, timeout_s=timeout_s)
                if not bool(rec["ok"]):
                    return CodeQLRobustnessResult(False, f"{err_tag}: {rec['error']}", 0.0, 0, 0, 0.0)

            num_edges = _count_csv_rows(edge_csv)
            num_nodes = _count_csv_rows(node_csv)
            density = 0.0 if num_nodes <= 0 else (num_edges / float(num_nodes))
            density_term = 1.0 - math.exp(-density)
            node_term = 1.0 - math.exp(-num_nodes / 120.0)
            scalar = _clip01(0.8 * density_term + 0.2 * node_term)

            return CodeQLRobustnessResult(
                ok=True,
                error="",
                scalar=float(scalar),
                num_nodes=int(num_nodes),
                num_edges=int(num_edges),
                density=float(density),
            )
    except Exception as exc:
        return CodeQLRobustnessResult(False, f"INTERNAL_ERROR: {exc}", 0.0, 0, 0, 0.0)
