"""Convert google-research/mbpp to verl parquet format.

Output schema:
    prompt         : list[dict] -- [{"role": "user", "content": problem + tests}]
    data_source    : "mbpp"
    reward_model   : {"style": "rule", "ground_truth": tests_json_string}
    ability        : "coding"
    extra_info     : {"index": int, "task_id": int, "split": str}

MBPP problems are short ("Write a function to sum two numbers...") with 3
unit-test asserts. The deepcoder reward function uses stdin/stdout, but
MBPP uses Python asserts -- check_correctness_mbpp from
recipe/pdpo/reward_score/mbpp_action_thought_reward.py runs them in a
local subprocess with a tight 6s timeout per problem (no sandbox needed).

Usage:
    python -m recipe.pdpo.data_preprocess.mbpp \
        --output_dir /workspace/data/mbpp \
        --train_max 400 --val_max 200
"""

from __future__ import annotations

import argparse
import json
import os

import datasets


_PROMPT_HEADER = (
    "You are an expert Python programmer. Solve the following problem. "
    "Return a complete Python implementation inside a single fenced code "
    "block (```python ... ```). Make sure the function name matches what "
    "the unit tests expect.\n\n"
)


def _row_to_verl(row: dict, index: int, split: str) -> dict:
    text = row.get("text") or row.get("prompt") or ""
    tests = row.get("test_list") or []
    if not isinstance(tests, list):
        tests = [tests]
    test_lines = "\n".join(str(t) for t in tests)

    # Show the unit tests to the model so it knows what API to implement.
    prompt_user = (
        _PROMPT_HEADER
        + f"Problem:\n{text}\n\n"
        + f"The solution must pass these tests:\n```python\n{test_lines}\n```"
    )

    return {
        "data_source": "mbpp",
        "prompt": [{"role": "user", "content": prompt_user}],
        "ability": "coding",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(tests)},
        "extra_info": {
            "index": index,
            "task_id": int(row.get("task_id", index)),
            "split": split,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/workspace/data/mbpp")
    p.add_argument("--train_max", type=int, default=400)
    p.add_argument("--val_max", type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # "full" has 374 train / 500 test / 90 val; "sanitized" is cleaner but smaller (120/257).
    print("Loading mbpp 'full' configuration")
    train_all = datasets.load_dataset("mbpp", "full", split="train")
    val_extra = datasets.load_dataset("mbpp", "full", split="validation")
    test = datasets.load_dataset("mbpp", "full", split="test")
    # Concat train + validation for a slightly bigger training set (464 problems).
    train = datasets.concatenate_datasets([train_all, val_extra])

    print(f"Train rows available: {len(train)}; test rows available: {len(test)}")
    if args.train_max and len(train) > args.train_max:
        train = train.select(range(args.train_max))
    if args.val_max and len(test) > args.val_max:
        test = test.select(range(args.val_max))

    train_rows = [_row_to_verl(row, i, "train") for i, row in enumerate(train)]
    val_rows = [_row_to_verl(row, i, "test") for i, row in enumerate(test)]

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "test.parquet")
    datasets.Dataset.from_list(train_rows).to_parquet(train_path)
    datasets.Dataset.from_list(val_rows).to_parquet(val_path)

    print(f"Wrote {len(train_rows)} train rows -> {train_path}")
    print(f"Wrote {len(val_rows)} val rows -> {val_path}")
    print("Sample row keys:", list(train_rows[0].keys()))
    print("Sample tests:", train_rows[0]["reward_model"]["ground_truth"])


if __name__ == "__main__":
    main()
