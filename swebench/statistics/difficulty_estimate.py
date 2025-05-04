import json
import re
from tqdm import tqdm
import concurrent.futures

import argparse

from datasets import load_dataset

from swebench.statistics.utils import call_api



def create_difficulty_prompt(instance):
    prompt = f"""
    You are a senior software engineer with over 10 years of solid experience in rust, cpp, python, and go. You possess a deep understanding of these languages and their standard libraries, along with a strong sense of problem difficulty.

    Your task is to evaluate the difficulty and clarity of a coding problem from a GitHub repository, given its "Problem Statement" and "Code Changes". You need to consider the following factors:

    1. **Clarity and complexity of the problem description:** Is the problem goal, input, output, and constraints clearly defined? Are there any ambiguities or missing critical details? Is the problem's logic inherently complex?
    2. **Scope and depth of code changes required to the whole codebase:** Does the modification involve a single file/function or multiple modules? Does it require understanding interactions between different parts of the codebase? What is the overall amount of code change? Does it impact the system's architecture?
    3. **Number of technical concepts that need to be understood:** What specific programming language features, libraries, algorithms, design patterns, or domain-specific knowledge are required to solve this problem? How complex are these concepts?
    4. **Potential edge cases and error handling requirements:** Does the problem statement mention any specific edge cases or error conditions to consider? Does the code change require adding or modifying error handling logic? How complex are these edge cases?

    Based on these factors, you will provide a Clarity Score and a Difficulty Score with detailed explanations.

    Here is the problem statement and code changes, delimited by '&&&':

    Problem Statement:
    &&&
    {instance["problem_statement"]}
    &&&

    Code Changes:
    &&&
    {instance["patch"]}
    &&&

    First, provide your judgment of the Clarity Scoring (0, 1, 2, 3) of the problem, along with your explanation:

    - 0 (Invalid): Statement is incomprehensible or code changes are unrelated.
    - 1 (Significant Ambiguities): Valid but lacks critical details (e.g., no input/output format).
    - 2 (Mostly Clear): Valid, clear, but minor details missing (e.g., edge cases not specified).
    - 3 (Comprehensive): Valid, clear, with detailed requirements and examples.

    Then, provide a difficulty score between 0.0 and 1.0, along with your explanation:

    - 0.0-0.2: Very easy, requires only basic code modifications (e.g., fixing a typo, changing a constant).
    - 0.2-0.4: Easy, requires understanding some code logic and making simple function or statement modifications (e.g., fixing a simple bug, adding a basic feature).
    - 0.4-0.6: Medium, requires understanding multiple concepts and making complex modifications across several files, potentially involving some edge case handling (e.g., implementing a new module with moderate complexity).
    - 0.6-0.8: Hard, requires deep understanding of the codebase architecture and complex modifications with significant impact, involving handling numerous edge cases and potential performance considerations (e.g., refactoring a core component, implementing a complex algorithm).
    - 0.8-1.0: Very hard, requires advanced technical knowledge, extensive experience, and tackling highly challenging problems with intricate logic, potentially involving system-level considerations or complex domain-specific knowledge (e.g., implementing a new distributed consensus protocol).

    Please return your response in the following structured format:
    <clarity_score>integer between 0 and 3</clarity_score>
    <clarity_explanation>Your explanation for the clarity score.</clarity_explanation>
    <difficulty>float between 0.00 and 1.00</difficulty>
    <difficulty_explanation>Your explanation for the difficulty score.</difficulty_explanation>
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
