# Note that the dataset is not included in the repository. We add it to the gitignore file.

import os
from datasets import load_dataset, load_from_disk
from pathlib import Path
from openai import OpenAI
from typing import Optional
import time

parquet_dir = Path("../../dataset/swing-bench/")
hf_dir = Path("../../dataset/swing-bench-hf-row")
filtered_hf_dir = Path("../../dataset/swing-bench-hf-filtered")


def build_dataset():

    if not filtered_hf_dir.exists():
        os.environ["HF_TOKEN"] = os.environ["SWINGBENCH_HF_TOKEN"]

        if not parquet_dir.exists():
            os.system(
                f"huggingface-cli download --repo-type dataset SwingBench/SwingBench-data --local-dir {parquet_dir}"
            )

        if not hf_dir.exists():
            dataset = load_dataset(
                "parquet",
                data_files={
                    "rust": str(parquet_dir / "data" / "rust-*.parquet"),
                    "cpp": str(parquet_dir / "data" / "cpp-*.parquet"),
                    "python": str(parquet_dir / "data" / "python-*.parquet"),
                    "go": str(parquet_dir / "data" / "go-*.parquet"),
                },
            )

            dataset.save_to_disk(hf_dir)
        else:
            dataset = load_from_disk(hf_dir)

        filtered_dataset = dataset.filter(
            lambda example: example["ci_name_list"] != []
        )

        filtered_dataset.save_to_disk(filtered_hf_dir)
    else:
        dataset = load_from_disk(filtered_hf_dir)

    print(dataset)
    return dataset


def call_api(
    prompt: str, api_key: str, base_url: str, model: str = "", max_attempts: int = 3
) -> Optional[str]:
    for attempt in range(max_attempts):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"API call attempt {attempt+1}/{max_attempts} failed: {str(e)}")
            if attempt < max_attempts - 1:
                time.sleep(2)

    return None