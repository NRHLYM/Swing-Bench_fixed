import json
import jsonlines
from typing import Dict, Any, List, Optional
import re
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import os
import time
import argparse

from datasets import load_dataset


def call_api(
    prompt: str, api_key: str, base_url: str, model: str = ""
) -> Optional[str]:
    for attempt in range(3):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"API call attempt {attempt+1}/3 failed: {str(e)}")
            if attempt < 2:
                time.sleep(2)

    return None


def create_difficulty_prompt(instance):
    prompt = f"""
    Please evaluate the difficulty of this coding problem based on the following factors:
    1. Clarity and complexity of the problem description
    2. Scope and depth of code changes required
    3. Number of technical concepts that need to be understood
    4. Complexity of the solution
    5. Potential edge cases and error handling requirements

    Problem Statement:
    {instance["problem_statement"]}

    Code Changes:
    {instance["patch"]}

    Please provide a difficulty score between 0 and 1, where:
    - 0.0-0.2: Very easy, requires only basic code modifications
    - 0.2-0.4: Easy, requires some code understanding and modifications
    - 0.4-0.6: Medium, requires understanding multiple concepts and complex modifications
    - 0.6-0.8: Hard, requires deep code understanding and complex modifications
    - 0.8-1.0: Very hard, requires advanced technical knowledge and complex solutions

    Please return in the following format:
    <difficulty>0.XX</difficulty>
    <explanation>Your explanation</explanation>
    """
    return prompt


def estimate_difficulty(instance, api_key, base_url, model):
    prompt = create_difficulty_prompt(instance)
    response = call_api(prompt, api_key, base_url, model)

    if not response:
        return None, "API call failed"

    try:
        difficulty_match = re.search(r"<difficulty>([0-9.]+)</difficulty>", response)
        if not difficulty_match:
            return None, "Could not parse difficulty score"

        difficulty = float(difficulty_match.group(1))

        explanation_match = re.search(
            r"<explanation>(.*?)</explanation>", response, re.DOTALL
        )
        explanation = (
            explanation_match.group(1)
            if explanation_match
            else "No explanation provided"
        )

        return difficulty, explanation
    except Exception as e:
        return None, f"Error parsing response: {str(e)}"


def process_instances_parallel(
    instances, output_file, api_key, base_url, model, num_workers=50
):
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(estimate_difficulty, instance, api_key, base_url, model)
            for instance in instances
        ]

        with tqdm(total=len(futures), desc="Evaluating difficulty") as progress:
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    difficulty, explanation = future.result()
                    if difficulty is not None:
                        result = {
                            "instance_id": instances[i].get("instance_id", i),
                            "difficulty": difficulty,
                            "explanation": explanation,
                        }
                        results.append(result)
                        with open(output_file, "a") as f:
                            f.write(json.dumps(result) + "\n")
                except Exception as e:
                    print(f"Error processing instance: {str(e)}")
                progress.update(1)

    return results


def analyze_difficulty_distribution(results):
    difficulties = [r["difficulty"] for r in results]
    avg_difficulty = sum(difficulties) / len(difficulties)

    print(f"\nDifficulty Analysis Results:")
    print(f"Total instances: {len(results)}")
    print(f"Average difficulty: {avg_difficulty:.3f}")

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins) - 1):
        count = sum(1 for d in difficulties if bins[i] <= d < bins[i + 1])
        percentage = (count / len(difficulties)) * 100
        print(f"Difficulty {bins[i]:.1f}-{bins[i+1]:.1f}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate difficulty of coding problems"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        default="/home/mnt/wdxu/github/SwingBench-data",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
        default="difficulty_results.jsonl",
    )
    parser.add_argument(
        "--api-key", type=str, help="API key for LLM service", default="no-api-key"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for LLM service",
        default="http://localhost:8000/v1/",
    )
    parser.add_argument(
        "--workers", type=int, help="Number of worker threads", default=16
    )
    parser.add_argument("--split", type=str, help="Split to process", default=None)
    parser.add_argument(
        "--languages", type=str, help="Languages to process", default="rust"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use",
        default="/home/mnt/wdxu/models/Qwen2.5-Coder-7B-Instruct",
    )
    args = parser.parse_args()

    language_list = args.languages.split(",")
    for language in language_list:
        dataset = load_dataset(args.dataset_name, split=args.split)[language]
        results = process_instances_parallel(
            dataset, args.output, args.api_key, args.base_url, args.model, args.workers
        )
        analyze_difficulty_distribution(results)
