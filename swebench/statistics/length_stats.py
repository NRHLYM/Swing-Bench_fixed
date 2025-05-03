from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

import argparse
import os
import json

fields = ['patch', 'test_patch', 'problem_statement', 'hints_text']


def get_length_bins(length):
    if length < 100:
        return "0-100"
    elif length < 250:
        return "100-250"
    elif length < 500:
        return "250-500"
    elif length < 750:
        return "500-750"
    elif length < 1000:
        return "750-1000"
    elif length < 1500:
        return "1000-1500"
    elif length < 2000:
        return "1500-2000"
    elif length < 2500:
        return "2000-2500"
    elif length < 3000:
        return "2500-3000"
    elif length < 3500:
        return "3000-3500"
    elif length < 5000:
        return "3500-5000"
    elif length < 7500:
        return "5000-7500"
    elif length < 10000:
        return "7500-10000"
    elif length < 15000:
        return "10000-15000"
    elif length < 20000:
        return "15000-20000"
    elif length < 25000:
        return "20000-25000"
    elif length < 30000:
        return "25000-30000"
    elif length < 35000:
        return "30000-35000"
    elif length < 50000:
        return "35000-50000"
    else:
        return "50000+"


def get_length_stats(language, dataset, tokenizer_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print('-' * 80)
    print(f"Language: {language}")
    print(f"Number of unsampled examples: {len(dataset)}")
    stats = {
        "language": language,
        "patch": {},
        "test_patch": {},
        "problem_statement": {},
        "hints_text": {}
    }
    dump_field_token_length_path = os.path.join(output_dir,
                                                f"{language}_field_token_length.data")
    with open(dump_field_token_length_path, "w") as f:
        for instance in tqdm(dataset, desc="Processing instances"):
            for field in fields:
                if field in instance and instance[field]:
                    token_length = len(tokenizer.encode(instance[field]))
                    bin_name = get_length_bins(token_length)
                    f.write(f"{instance['repo']} {instance['instance_id']} {field} {token_length}\n")

                    if bin_name not in stats[field]:
                        stats[field][bin_name] = 0
                    stats[field][bin_name] += 1

    with open(os.path.join(output_dir, f"{language}_summary.json"), "w") as f:
        json.dump(stats, f)


def main(**kwargs):
    output_dir = kwargs["output_dir"]
    data_path = kwargs["data_path"]
    tokenizer_path = kwargs["tokenizer_path"]
    language_list = kwargs["language_list"].split(",")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Number of unsampled examples: ", end="")
    for language in language_list:
        dataset = load_dataset(data_path)[language]
        get_length_stats(language, dataset, tokenizer_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/mnt/wdxu/github/SwingBench-data")
    parser.add_argument("--language_list", type=str, default="rust,cpp,python,go")
    parser.add_argument("--tokenizer_path", type=str, default="/home/mnt/wdxu/models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./length_stats")
    args = parser.parse_args()

    main(**vars(args))
