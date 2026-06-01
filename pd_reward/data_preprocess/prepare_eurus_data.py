import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
REPO_DIR = PROJECT_DIR.parent
for path in (PROJECT_DIR, REPO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reward_score.coding_executable_reward import parse_io_tests


EURUS_DATASET = "PRIME-RL/Eurus-2-RL-Data"
CODING_SOURCES = {
    "apps",
    "code_contests",
    "codecontests",
    "codeforces",
    "taco",
}
EURUS_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field(
            "prompt",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string()),
                        pa.field("content", pa.large_string()),
                    ]
                )
            ),
        ),
        pa.field("ability", pa.string()),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("style", pa.string()),
                    pa.field("ground_truth", pa.large_string()),
                ]
            ),
        ),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("index", pa.int64()),
                    pa.field("split", pa.string()),
                ]
            ),
        ),
    ]
)


def _source_key(data_source: Any) -> str:
    return str(data_source or "").strip().lower().replace("-", "_")


def is_eurus_coding_row(row: dict[str, Any]) -> bool:
    ability = str(row.get("ability") or "").strip().lower()
    data_source = _source_key(row.get("data_source"))
    if ability in {"code", "coding"}:
        return True
    return data_source in CODING_SOURCES


def _normalized_reward_model(reward_model: Any) -> dict[str, Any] | None:
    if not isinstance(reward_model, dict):
        return None

    parsed = parse_io_tests({"tests": reward_model.get("ground_truth")})
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
    normalized["style"] = str(normalized.get("style") or "rule")
    normalized["ground_truth"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "style": normalized["style"],
        "ground_truth": normalized["ground_truth"],
    }


def _text_or_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalized_prompt(prompt: Any) -> list[dict[str, str]] | None:
    messages: list[dict[str, str]] = []
    if isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict):
                role = _text_or_json(item.get("role")).strip() or "user"
                content = _text_or_json(item.get("content"))
            else:
                role = "user"
                content = _text_or_json(item)
            if content.strip():
                messages.append({"role": role, "content": content})
    else:
        content = _text_or_json(prompt)
        if content.strip():
            messages.append({"role": "user", "content": content})

    return messages or None


def _coerce_index(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _canonical_eurus_row(
    row: dict[str, Any], *, output_index: int, split: str | None
) -> tuple[dict[str, Any] | None, str | None]:
    reward_model = _normalized_reward_model(row.get("reward_model"))
    if reward_model is None:
        return None, "invalid_tests"

    prompt = _normalized_prompt(row.get("prompt"))
    if prompt is None:
        return None, "invalid_prompt"

    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    index = _coerce_index(extra_info.get("index", row.get("index")), output_index)
    split_value = _text_or_json(extra_info.get("split", row.get("split", split))).strip()

    return (
        {
            "data_source": _text_or_json(row.get("data_source") or "eurus").strip() or "eurus",
            "prompt": prompt,
            "ability": _text_or_json(row.get("ability") or "code").strip() or "code",
            "reward_model": reward_model,
            "extra_info": {
                "index": index,
                "split": split_value,
            },
        },
        None,
    )


def prepare_eurus_rows(rows: Iterable[dict[str, Any]], split: str | None = None) -> tuple[list[dict[str, Any]], Counter[str]]:
    cleaned = []
    dropped = Counter()
    for row in rows:
        if not is_eurus_coding_row(row):
            dropped["non_coding"] += 1
            continue

        normalized_row, drop_reason = _canonical_eurus_row(row, output_index=len(cleaned), split=split)
        if normalized_row is None:
            dropped[drop_reason or "invalid_row"] += 1
            continue

        cleaned.append(normalized_row)
    return cleaned, dropped


def _write_parquet_rows(rows: list[dict[str, Any]], path: Path, row_group_size: int = 256) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    if row_group_size <= 0:
        raise ValueError(f"row_group_size must be positive, got {row_group_size}")

    writer = None
    try:
        for start in range(0, len(rows), row_group_size):
            table = pa.Table.from_pylist(rows[start : start + row_group_size], schema=EURUS_PARQUET_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _load_split(dataset_name: str, split: str, local_dataset_path: str | None = None):
    import datasets

    source = local_dataset_path or dataset_name
    dataset = datasets.load_dataset(source)
    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise KeyError(f"Split {split!r} not found in {source}; available splits: {available}")
    return dataset[split]


def _prepare_split(
    *,
    dataset_name: str,
    split: str,
    output_path: Path,
    local_dataset_path: str | None,
) -> None:
    rows = _load_split(dataset_name, split, local_dataset_path=local_dataset_path)
    input_rows = len(rows)
    cleaned, dropped = prepare_eurus_rows(rows, split=split)
    _write_parquet_rows(cleaned, output_path)
    print(f"split={split}")
    print(f"input_rows={input_rows}")
    print(f"output_rows={len(cleaned)}")
    print(f"dropped_rows={sum(dropped.values())}")
    for reason, count in sorted(dropped.items()):
        print(f"dropped[{reason}]={count}")
    print(f"saved={output_path}")


def main() -> None:
    default_data_home = os.environ.get("RAY_DATA_HOME", "/shared/nas2/yujiz/rl/data")
    parser = argparse.ArgumentParser(description="Prepare Eurus-2-RL coding train/eval parquet files.")
    parser.add_argument("--dataset", default=EURUS_DATASET, help="Hugging Face dataset name")
    parser.add_argument("--local-dataset-path", default=None, help="Optional local dataset path for datasets.load_dataset")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument(
        "--train-output",
        default=os.path.join(default_data_home, "eurus", "eurus_code_train.parquet"),
    )
    parser.add_argument(
        "--val-output",
        default=os.path.join(default_data_home, "eurus", "eurus_code_val.parquet"),
    )
    args = parser.parse_args()

    _prepare_split(
        dataset_name=args.dataset,
        split=args.train_split,
        output_path=Path(args.train_output).expanduser(),
        local_dataset_path=args.local_dataset_path,
    )
    _prepare_split(
        dataset_name=args.dataset,
        split=args.val_split,
        output_path=Path(args.val_output).expanduser(),
        local_dataset_path=args.local_dataset_path,
    )


if __name__ == "__main__":
    main()
