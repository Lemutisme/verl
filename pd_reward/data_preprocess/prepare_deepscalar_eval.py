import argparse
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
DEFAULT_OLYMPIAD_DATASET = "Hothan/OlympiadBench"
DEFAULT_OLYMPIAD_CONFIG = "OE_TO_maths_en_COMP"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _stringify(value[0])
        return "\n".join(_stringify(item) for item in value if _stringify(item))
    if isinstance(value, dict):
        for key in ("text", "content", "question", "problem", "answer", "final_answer"):
            if key in value:
                text = _stringify(value[key])
                if text:
                    return text
        return " ".join(_stringify(item) for item in value.values() if _stringify(item))
    return str(value)


def _with_hash_instruction(question: str) -> str:
    text = question.strip()
    if "####" in text and "output the final answer" in text.lower():
        return text
    return text + " " + DEFAULT_INSTRUCTION


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    parquet_file = pq.ParquetFile(path)
    rows: list[dict[str, Any]] = []
    for row_group in range(parquet_file.num_row_groups):
        rows.extend(parquet_file.read_row_group(row_group).to_pylist())
    return rows


def _canonical_eval_row(
    *,
    data_source: str,
    question: str,
    answer: str,
    split: str,
    index: int,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = dict(extra_info or {})
    extra.setdefault("split", split)
    extra.setdefault("index", index)
    extra.setdefault("question", question)
    extra.setdefault("answer", answer)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": _with_hash_instruction(question)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra,
    }


def _canonical_existing_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    reward_model = row.get("reward_model")
    prompt = row.get("prompt")
    if not isinstance(reward_model, dict) or prompt is None:
        return None

    data_source = str(row.get("data_source") or "unknown")
    ground_truth = _stringify(reward_model.get("ground_truth"))
    if not ground_truth:
        return None

    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        question = _stringify(prompt[0].get("content"))
    else:
        question = _stringify(prompt)
    if not question:
        return None

    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return _canonical_eval_row(
        data_source=data_source,
        question=question,
        answer=ground_truth,
        split=str(extra.get("split") or "eval"),
        index=index,
        extra_info=extra,
    )


def _olympiad_question_answer(row: dict[str, Any]) -> tuple[str, str]:
    question = ""
    for key in ("question", "problem", "prompt"):
        question = _stringify(row.get(key))
        if question:
            break

    answer = ""
    for key in ("final_answer", "answer", "solution"):
        answer = _stringify(row.get(key))
        if answer:
            break

    return question, answer


def _canonical_olympiad_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for idx, row in enumerate(rows):
        question, answer = _olympiad_question_answer(row)
        if not question or not answer:
            continue
        extra = {
            "split": row.get("split", "test"),
            "index": idx,
        }
        for key in ("subfield", "subject", "answer_type", "language"):
            if key in row:
                extra[key] = row[key]
        normalized.append(
            _canonical_eval_row(
                data_source="olympiad",
                question=question,
                answer=answer,
                split=str(extra["split"]),
                index=idx,
                extra_info=extra,
            )
        )
    return normalized


def build_deepscalar_eval_rows(
    base_rows: list[dict[str, Any]],
    general365_rows: list[dict[str, Any]],
    olympiad_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(base_rows):
        normalized = _canonical_existing_row(row, idx)
        if normalized is not None:
            rows.append(normalized)

    offset = len(rows)
    for idx, row in enumerate(general365_rows):
        normalized = _canonical_existing_row(row, offset + idx)
        if normalized is not None:
            normalized["data_source"] = "general365"
            rows.append(normalized)

    rows.extend(_canonical_olympiad_rows(olympiad_rows))
    return rows


def _load_olympiad_rows(dataset_name: str, config: str, split: str) -> list[dict[str, Any]]:
    import datasets

    dataset = datasets.load_dataset(dataset_name, config)
    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise KeyError(f"Split {split!r} not found in {dataset_name}/{config}; available: {available}")
    return [dict(row, split=split) for row in dataset[split]]


def _write_parquet_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)


def main() -> None:
    default_data_home = os.environ.get("RAY_DATA_HOME", "/shared/nas2/yujiz/rl/data")
    parser = argparse.ArgumentParser(description="Build the DeepScaleR math eval suite.")
    parser.add_argument(
        "--base-eval",
        default=os.path.join(default_data_home, "math", "math_eval_master.parquet"),
        help="Existing math eval parquet, e.g. AIME/AMC/MATH500/Minerva.",
    )
    parser.add_argument(
        "--general365-eval",
        default=os.path.join(default_data_home, "general365", "test_formatted.parquet"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(default_data_home, "math", "math_eval_deepscalar.parquet"),
    )
    parser.add_argument("--olympiad-dataset", default=DEFAULT_OLYMPIAD_DATASET)
    parser.add_argument("--olympiad-config", default=DEFAULT_OLYMPIAD_CONFIG)
    parser.add_argument("--olympiad-split", default="train")
    args = parser.parse_args()

    base_rows = _read_parquet_rows(Path(args.base_eval).expanduser())
    general365_rows = _read_parquet_rows(Path(args.general365_eval).expanduser())
    olympiad_rows = _load_olympiad_rows(args.olympiad_dataset, args.olympiad_config, args.olympiad_split)
    rows = build_deepscalar_eval_rows(base_rows, general365_rows, olympiad_rows)
    _write_parquet_rows(rows, Path(args.output).expanduser())

    print(f"base_rows={len(base_rows)}")
    print(f"general365_rows={len(general365_rows)}")
    print(f"olympiad_rows={len(olympiad_rows)}")
    print(f"output_rows={len(rows)}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
