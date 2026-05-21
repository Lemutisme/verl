import argparse
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
BOXED_FORMAT_BLOCK_RE = re.compile(
    r"\s*Output your final answer at the end of your reply using the following format:\s*"
    r"### The final answer is:\s*\$?\\boxed\{<Your Answer>\}\$?\s*"
    r"For example:\s*"
    r"### The final answer is:\s*\$?\\boxed\{123\}\$?",
    re.IGNORECASE,
)
BOXED_INSTRUCTION_RE = re.compile(
    r"Let's think step by step and output the final answer within\s+\\boxed\{\}\.",
    re.IGNORECASE,
)


def _format_content(content: str, instruction: str, force_hash: bool) -> str:
    if force_hash:
        content = BOXED_FORMAT_BLOCK_RE.sub("", content).strip()
        content = BOXED_INSTRUCTION_RE.sub(instruction, content).strip()
        lowered = content.lower()
        if "####" not in content or "output the final answer" not in lowered:
            content = content.rstrip() + "\n" + instruction
        return content

    lowered = content.lower()
    if "output the final answer" in lowered or "the final answer is" in lowered:
        return content
    return content.rstrip() + "\n" + instruction


def _format_prompt(prompt: Any, instruction: str, force_hash: bool) -> Any:
    if isinstance(prompt, list):
        formatted = []
        updated = False
        for item in prompt:
            if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str):
                content = item["content"]
                new_content = _format_content(content, instruction, force_hash)
                if new_content != content:
                    item = dict(item)
                    item["content"] = new_content
                    updated = True
            formatted.append(item)
        if updated:
            return formatted
        return prompt

    if isinstance(prompt, str):
        return _format_content(prompt, instruction, force_hash)

    return prompt


def _read_parquet_rows(path: Path) -> tuple[list[dict[str, Any]], pa.Schema]:
    parquet_file = pq.ParquetFile(path)
    rows: list[dict[str, Any]] = []
    for row_group in range(parquet_file.num_row_groups):
        rows.extend(parquet_file.read_row_group(row_group).to_pylist())
    return rows, parquet_file.schema_arrow


def _write_parquet_rows(rows: list[dict[str, Any]], schema: pa.Schema, path: Path, batch_size: int = 1024) -> None:
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


def format_math_rows(
    rows: list[dict[str, Any]],
    instruction: str = DEFAULT_INSTRUCTION,
    force_hash: bool = False,
) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        row = dict(row)
        row["prompt"] = _format_prompt(row.get("prompt"), instruction, force_hash)
        formatted.append(row)
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a final-answer format instruction to math prompts.")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--force-hash",
        action="store_true",
        help="Normalize existing boxed final-answer instructions to the #### format.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    rows, schema = _read_parquet_rows(input_path)
    formatted = format_math_rows(rows, args.instruction, force_hash=args.force_hash)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_rows(formatted, schema, output_path)
    print(f"input_rows={len(rows)}")
    print(f"output_rows={len(formatted)}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
