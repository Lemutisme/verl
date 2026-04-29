import argparse
import json
import os
import datasets
from verl.utils.hdfs_io import copy, makedirs

def extract_solution(answer):
    if isinstance(answer, str) and "\\boxed{" in answer:
        # Extract content from \boxed{}
        start = answer.find("\\boxed{") + 7
        # Find matching closing brace
        brace_count = 1
        end = start
        while end < len(answer) and brace_count > 0:
            if answer[end] == '{':
                brace_count += 1
            elif answer[end] == '}':
                brace_count -= 1
            end += 1
        if brace_count == 0:
            return answer[start:end-1]
    return str(answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_save_dir", default="~/data/general365", help="Save directory")
    args = parser.parse_args()

    data_source = "meituan-longcat/General365_Public"
    print(f"Loading {data_source}...")
    
    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    # General365 only has a 'test' split. We will use it for both train and test 
    # or just split it 90/10 for demonstration since it's only 365 items.
    full_dataset = dataset["test"].shuffle(seed=42)
    split_idx = int(len(full_dataset) * 0.9)
    train_dataset = full_dataset.select(range(split_idx))
    test_dataset = full_dataset.select(range(split_idx, len(full_dataset)))

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            question = question + " " + instruction_following
            
            raw_answer = example.pop("answer")
            solution = extract_solution(raw_answer)
            
            data = {
                "data_source": "general365",
                "prompt": [{"role": "user", "content": question}],
                "ability": "reasoning",
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
