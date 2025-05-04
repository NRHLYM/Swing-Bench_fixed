# Note that the dataset is not included in the repository. We add it to the gitignore file.

import os
from pathlib import Path
from datasets import load_dataset, load_from_disk, DatasetDict

parquet_dir = Path("../../dataset/swing-bench/")
hf_dir = Path("../../dataset/swing-bench-hf-row")
filtered_hf_dir = Path("../../dataset/swing-bench-hf-filtered")

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
