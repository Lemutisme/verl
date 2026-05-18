import os
import pandas as pd
import datasets

DATASETS_TO_LOAD = {
    'aime24': {'path': 'AI-MO/aimo-validation-aime'},
    'aime25': {'path': 'math-ai/aime25'},
    'amc': {'path': 'AI-MO/aimo-validation-amc'},
    'math500': {'path': 'HuggingFaceH4/MATH-500'},
    'minerva_math': {'path': 'math-ai/minervamath'},
    'olympiad': {'path': 'GAIR/OlympiadBench', 'config': 'TOE_maths_en_COMP'}
}

instruction_following = 'Let\'s think step by step and output the final answer after "####".'

all_records = []

for ds_name, info in DATASETS_TO_LOAD.items():
    print(f"Loading {ds_name}...")
    try:
        config = info.get('config', 'default')
        if config != 'default':
            ds = datasets.load_dataset(info['path'], config, trust_remote_code=True)
        else:
            # Fallback to default
            try:
                ds = datasets.load_dataset(info['path'], trust_remote_code=True)
            except ValueError:
                # If it requires a config name but default doesn't work
                ds = datasets.load_dataset(info['path'], 'default', trust_remote_code=True)
        
        split = 'test' if 'test' in ds else ('validation' if 'validation' in ds else 'train')
        split_ds = ds[split]
        
        count = 0
        for i, row in enumerate(split_ds):
            # Try to find question
            question = row.get('problem', row.get('question', ''))
            
            # Try to find answer
            answer = row.get('answer', row.get('final_answer', row.get('solution', '')))
            
            if not question:
                continue
                
            # Append prompt instructions like verl gsm8k preprocessing does, so that we
            # encourage the model to output #### format.
            question_with_instr = question + " " + instruction_following
                
            record = {
                "data_source": ds_name,
                "prompt": [{"role": "user", "content": question_with_instr}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {"split": split, "index": i, "answer": str(answer), "question": question}
            }
            all_records.append(record)
            count += 1
            
        print(f"Added {count} records from {ds_name}.")
    except Exception as e:
        print(f"Failed to load {ds_name}: {e}")

df = pd.DataFrame(all_records)
print(f"Total records: {len(df)}")
os.makedirs('/shared/nas2/yujiz/rl/data/math', exist_ok=True)
output_file = '/shared/nas2/yujiz/rl/data/math/math_eval_master.parquet'
df.to_parquet(output_file)
print(f"Saved to {output_file}")
