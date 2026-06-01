from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_prepare_data_runs_all_reward_data_repairs():
    script = (PROJECT_DIR / "data_preprocess" / "prepare_data.sh").read_text()

    assert "format_math_prompts.py" in script
    assert "deepscalar_train_formatted.parquet" in script
    assert "general365/train_formatted.parquet" in script
    assert "general365/test_formatted.parquet" in script
    assert "prepare_deepscalar_eval.py" in script
    assert "math_eval_deepscalar.parquet" in script
    assert "--force-hash" in script
    assert "prepare_eurus_data.py" in script
    assert "eurus_code_train.parquet" in script
    assert "eurus_code_val.parquet" in script
    assert "clean_deepcoder_data.py" not in script


def test_eurus_prepare_filters_to_valid_coding_rows():
    from data_preprocess.prepare_eurus_data import prepare_eurus_rows

    rows = [
        {
            "data_source": "code_contests",
            "ability": "code",
            "prompt": [{"role": "user", "content": "solve"}],
            "reward_model": {"style": "rule", "ground_truth": '{"inputs": ["1"], "outputs": ["1"]}'},
            "extra_info": {"index": 0},
        },
        {
            "data_source": "math",
            "ability": "math",
            "prompt": [{"role": "user", "content": "math"}],
            "reward_model": {"style": "rule", "ground_truth": "42"},
            "extra_info": {"index": 1},
        },
        {
            "data_source": "taco",
            "ability": "code",
            "prompt": [{"role": "user", "content": "bad"}],
            "reward_model": {"style": "rule", "ground_truth": "{}"},
            "extra_info": {"index": 2},
        },
    ]

    cleaned, dropped = prepare_eurus_rows(rows)

    assert len(cleaned) == 1
    assert cleaned[0]["data_source"] == "code_contests"
    assert set(cleaned[0]) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert dropped["non_coding"] == 1
    assert dropped["invalid_tests"] == 1


def test_eurus_prepare_projects_rows_to_verl_compatible_schema():
    from data_preprocess.prepare_eurus_data import prepare_eurus_rows

    rows = [
        {
            "data_source": "taco",
            "ability": "code",
            "prompt": [{"role": "user", "content": "solve"}],
            "reward_model": {"style": "rule", "ground_truth": '{"inputs": ["1"], "outputs": ["1"]}'},
            "extra_info": {
                "index": "7",
                "split": "train",
                "nested_metadata": {"large": ["value"]},
            },
            "messages": [{"deep": {"unused": True}}],
        }
    ]

    cleaned, dropped = prepare_eurus_rows(rows, split="train")

    assert dropped == {}
    assert cleaned == [
        {
            "data_source": "taco",
            "prompt": [{"role": "user", "content": "solve"}],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": '{"inputs":["1"],"outputs":["1"]}',
            },
            "extra_info": {"index": 7, "split": "train"},
        }
    ]


def test_eurus_parquet_loads_with_hf_datasets(tmp_path):
    import datasets

    from data_preprocess.prepare_eurus_data import _write_parquet_rows, prepare_eurus_rows

    source_rows = []
    for idx in range(3):
        source_rows.append(
            {
                "data_source": "code_contests",
                "ability": "code",
                "prompt": [{"role": "user", "content": f"solve {idx}"}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": '{"inputs": ["1"], "outputs": ["1"]}',
                },
                "extra_info": {"index": idx, "split": "train"},
                "unused_nested_field": {"chunky": [{"value": idx}]},
            }
        )

    rows, dropped = prepare_eurus_rows(source_rows, split="train")
    output_path = tmp_path / "eurus_code_train.parquet"
    _write_parquet_rows(rows, output_path, row_group_size=1)

    assert dropped == {}
    assert pq.ParquetFile(output_path).num_row_groups == 3
    parquet_file = pq.ParquetFile(output_path)
    assert parquet_file.schema_arrow.field("prompt").type.value_type.field("content").type == pa.large_string()
    assert parquet_file.schema_arrow.field("reward_model").type.field("ground_truth").type == pa.large_string()
    assert parquet_file.read(columns=["reward_model"]).num_rows == 3
    assert parquet_file.read(columns=["reward_model.ground_truth"]).num_rows == 3

    dataset = datasets.load_dataset(
        "parquet",
        data_files=str(output_path),
        split="train",
        cache_dir=str(tmp_path / "datasets_cache"),
    )

    assert len(dataset) == 3
    first = dataset[0]
    assert first["prompt"] == [{"role": "user", "content": "solve 0"}]
    assert first["reward_model"]["ground_truth"] == '{"inputs":["1"],"outputs":["1"]}'
    assert first["extra_info"] == {"index": 0, "split": "train"}


def test_deepscalar_eval_builder_combines_base_general365_and_olympiad_rows():
    from data_preprocess.prepare_deepscalar_eval import build_deepscalar_eval_rows

    base_rows = [
        {
            "data_source": "aime24",
            "prompt": [{"role": "user", "content": "base ####"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "1"},
            "extra_info": {"index": 0},
        }
    ]
    general365_rows = [
        {
            "data_source": "general365",
            "prompt": [{"role": "user", "content": "general ####"}],
            "ability": "reasoning",
            "reward_model": {"style": "rule", "ground_truth": "2"},
            "extra_info": {"index": 0},
        }
    ]
    olympiad_rows = [
        {
            "question": "olympiad problem",
            "final_answer": "\\frac{1}{2}",
            "answer_type": "Numerical",
            "subfield": "Geometry",
        }
    ]

    rows = build_deepscalar_eval_rows(base_rows, general365_rows, olympiad_rows)

    assert [row["data_source"] for row in rows] == ["aime24", "general365", "olympiad"]
    assert rows[-1]["reward_model"]["ground_truth"] == "\\frac{1}{2}"
    assert 'after "####"' in rows[-1]["prompt"][0]["content"]
