import os
import pyarrow.parquet as pq
import pandas as pd

MAX_ROWS = 200

# 1. Read deepcoder_full_val
df_deepcoder = pd.read_parquet('/shared/nas2/yujiz/rl/data/math/deepcoder_full_val.parquet')
print(f"Loaded {len(df_deepcoder)} from deepcoder_full_val")
all_dfs = [df_deepcoder]

# 2. Read APPS
try:
    pf_apps = pq.ParquetFile('/shared/nas2/yujiz/rl/data/coding/apps_intro_interview_test.parquet')
    df_apps = pf_apps.read_row_group(0).to_pandas().head(MAX_ROWS)
    # Ensure data_source is mapped to start with deepcoder_
    df_apps['data_source'] = 'deepcoder_apps'
    all_dfs.append(df_apps)
    print(f"Loaded {len(df_apps)} from APPS")
except Exception as e:
    print("Failed to load APPS:", e)

# 3. Read LCB
try:
    pf_lcb = pq.ParquetFile('/shared/nas2/yujiz/rl/data/coding/livecodebench.parquet')
    df_lcb = pf_lcb.read_row_group(0).to_pandas().head(MAX_ROWS)
    df_lcb['data_source'] = 'deepcoder_lcb'
    all_dfs.append(df_lcb)
    print(f"Loaded {len(df_lcb)} from LCB")
except Exception as e:
    print("Failed to load LCB:", e)

df_out = pd.concat(all_dfs, ignore_index=True)
print(f"Total code records: {len(df_out)}")

os.makedirs('/shared/nas2/yujiz/rl/data/coding', exist_ok=True)
output_file = '/shared/nas2/yujiz/rl/data/coding/code_eval_master.parquet'
df_out.to_parquet(output_file)
print(f"Saved to {output_file}")
