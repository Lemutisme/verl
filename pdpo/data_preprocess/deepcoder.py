"""Convert agentica-org/DeepCoder-Preview-Dataset to verl-style parquet.

Output schema matches what verl's NaiveRewardManager + our custom_reward expect:
    prompt         : list[dict] -- [{"role": "user", "content": problem}]
    data_source    : "deepcoder"
    reward_model   : {"style": "rule", "ground_truth": tests_json_string}
    ability        : "coding"
    extra_info     : {"index": int, "subset": str}

``tests`` is stored as a JSON string in reward_model.ground_truth. The
deepcoder reward function (recipe/pdpo/reward_score/deepcoder_action_thought_reward.py)
reads it back via _get_tests_deepcoder.

Usage:
    python -m recipe.pdpo.data_preprocess.deepcoder \
        --output_dir /workspace/data/deepcoder \
        --train_subset taco --train_max 2000 \
        --val_subset codeforces
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import datasets


_PROMPT_HEADER = (
    "Solve the following programming problem. Write a complete Python program. "
    "Place the solution inside a single fenced code block (```python ... ```). "
    "The program should read input via stdin and write output via stdout.\n\n"
    "Problem:\n"
)


def _coerce_tests(raw: Any) -> str:
    """Normalize the tests field into a JSON string of inputs/outputs."""
    if raw is None:
        return "{}"
    if isinstance(raw, str):
        # Already a JSON string (some subsets store it this way).
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            return raw
    if isinstance(raw, list):
        return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    return json.dumps(raw, ensure_ascii=False)


def _row_to_verl(row: dict, index: int, subset: str) -> dict:
    problem = row.get("problem") or row.get("question") or row.get("prompt") or ""
    tests = _coerce_tests(row.get("tests"))
    return {
        "data_source": "deepcoder",
        "prompt": [{"role": "user", "content": _PROMPT_HEADER + str(problem)}],
        "ability": "coding",
        "reward_model": {"style": "rule", "ground_truth": tests},
        "extra_info": {"index": index, "subset": subset, "split": "train"},
    }


_SUBSET_SPLIT = {
    "taco": "train",
    "primeintellect": "train",
    "lcbv5": "train",
    "codeforces": "test",  # only split this subset ships with
}


def _build(subset: str, max_rows: int | None, seed: int = 42) -> list[dict]:
    split = _SUBSET_SPLIT.get(subset, "train")
    ds = datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", subset, split=split)
    n = len(ds)
    if max_rows is not None and n > max_rows:
        random.seed(seed)
        idxs = random.sample(range(n), max_rows)
        idxs.sort()
        ds = ds.select(idxs)
    return [_row_to_verl(row, i, subset) for i, row in enumerate(ds)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/workspace/data/deepcoder")
    p.add_argument("--train_subset", default="taco", choices=["taco", "primeintellect", "codeforces", "lcbv5"])
    p.add_argument("--val_subset", default="codeforces", choices=["taco", "primeintellect", "codeforces", "lcbv5"])
    p.add_argument("--train_max", type=int, default=2000)
    p.add_argument("--val_max", type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building train from subset={args.train_subset} max={args.train_max}")
    train_rows = _build(args.train_subset, args.train_max)
    print(f"Building val from subset={args.val_subset} max={args.val_max}")
    val_rows = _build(args.val_subset, args.val_max)

    # Save with a stable schema: switch split field for val rows.
    for r in val_rows:
        r["extra_info"]["split"] = "val"

    train_ds = datasets.Dataset.from_list(train_rows)
    val_ds = datasets.Dataset.from_list(val_rows)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"Wrote {len(train_rows)} train rows -> {train_path}")
    print(f"Wrote {len(val_rows)} val rows -> {val_path}")
    print("Sample train row keys:", list(train_rows[0].keys()))
    print("Sample tests (truncated):", str(train_rows[0]['reward_model']['ground_truth'])[:200])


if __name__ == "__main__":
    main()
