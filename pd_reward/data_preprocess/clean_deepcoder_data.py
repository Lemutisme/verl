import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from reward_score.deepcoder_action_thought_reward import _get_tests_deepcoder


def _normalized_reward_model(reward_model: Any) -> dict[str, Any] | None:
    if not isinstance(reward_model, dict):
        return None

    ground_truth = reward_model.get("ground_truth")
    parsed = _get_tests_deepcoder({"tests": ground_truth})
    inputs = parsed.get("inputs") or []
    outputs = parsed.get("outputs") or []
    if not inputs or len(inputs) != len(outputs):
        return None

    payload: dict[str, Any] = {
        "inputs": inputs,
        "outputs": outputs,
    }
    if parsed.get("fn_name"):
        payload["fn_name"] = parsed["fn_name"]
    assert_cases = parsed.get("assert_case") or []
    if any(str(case).strip() for case in assert_cases):
        payload["assert_case"] = assert_cases

    normalized = dict(reward_model)
    normalized["ground_truth"] = json.dumps(payload, ensure_ascii=False)
    return normalized


def _read_parquet_rows(path: Path) -> tuple[list[dict[str, Any]], pa.Schema]:
    parquet_file = pq.ParquetFile(path)
    rows: list[dict[str, Any]] = []
    for row_group in range(parquet_file.num_row_groups):
        rows.extend(parquet_file.read_row_group(row_group).to_pylist())
    return rows, parquet_file.schema_arrow


def _write_parquet_rows(rows: list[dict[str, Any]], schema: pa.Schema, path: Path, batch_size: int = 512) -> None:
    writer = None
    try:
        for start in range(0, len(rows), batch_size):
            table = pa.Table.from_pylist(rows[start : start + batch_size], schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def clean_deepcoder_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept_rows = []
    dropped = Counter()
    for row in rows:
        reward_model = _normalized_reward_model(row.get("reward_model"))
        data_source = str(row.get("data_source") or "unknown")
        if reward_model is None:
            dropped[data_source] += 1
            continue
        row = dict(row)
        row["reward_model"] = reward_model
        kept_rows.append(row)
    return kept_rows, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and filter DeepCoder parquet files.")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    rows, schema = _read_parquet_rows(input_path)
    cleaned, dropped = clean_deepcoder_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_rows(cleaned, schema, output_path)

    print(f"input_rows={len(rows)}")
    print(f"output_rows={len(cleaned)}")
    print(f"dropped_rows={sum(dropped.values())}")
    for data_source, count in sorted(dropped.items()):
        print(f"dropped[{data_source}]={count}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
