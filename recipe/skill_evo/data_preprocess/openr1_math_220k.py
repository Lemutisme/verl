import argparse
import json
import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_save_dir", default="~/data/openr1_math", help="Save directory")
    args = parser.parse_args()

    data_source = "open-r1/OpenR1-Math-220k"
    print(f"Loading {data_source}...")
    
    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(data_source, "default")

    # OpenR1-Math-220k has 'train' split. Let's split 95/5 for train/test
    full_dataset = dataset["train"]
    # Since it's huge, doing a train_test_split is convenient
    split_dataset = full_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            question = question + " " + instruction_following
            
            # The dataset has 'answer' as the ground truth already, but if missing, extract from solution
            if "answer" in example and example["answer"] is not None:
                solution = str(example.pop("answer"))
            else:
                raw_solution = example.pop("solution")
                solution = extract_solution(raw_solution)
            
            data = {
                "data_source": "openr1_math_220k",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir if args.local_dir else args.local_save_dir
    local_dir = os.path.expanduser(local_save_dir)

    makedirs(local_dir)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)
    
    print(f"Preprocessed data saved to {local_dir}")
