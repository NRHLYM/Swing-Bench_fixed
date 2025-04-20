import json
import jsonlines
from typing import Dict, Any, List, Optional
import re
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import os
import time

def get_llm(prompt: str, api_key: str, base_url: str) -> Optional[str]:
    """Call LLM API with basic retry logic"""
    for attempt in range(3):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="public-glm-4-plus",
                temperature=0.7,
                top_p=0.8,
                stream=False,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call attempt {attempt+1}/3 failed: {str(e)}")
            if attempt < 2:
                time.sleep(2)
    
    return None

def create_vague_problem_statement_prompt(instance):
    prompt = f"""
    Evaluate if this problem statement is vague based on these criteria:
    1. Missing specificity about what to implement/fix
    2. Insufficient context or background
    3. Unclear requirements
    4. Undefined scope/boundaries
    5. Ambiguous technical specifications
    6. Lacks actionable information for a developer
    7. Contains contradictions or inconsistencies

    Problem Statement: {instance["problem_statement"]}

    Return your judgment as: <judgement>True</judgement> if vague, or <judgement>False</judgement> if not vague.
    """
    return prompt

def is_vague_problem_statement(instance, api_key, base_url):
    prompt = create_vague_problem_statement_prompt(instance)
    response = get_llm(prompt, api_key, base_url)
    
    if not response:
        return True  # Consider vague if API fails
        
    try:
        match = re.search(r'<judgement>(True|False)</judgement>', response, re.IGNORECASE)
        if match:
            result = match.group(1).lower()
            return result == 'true'

        if 'true' in response.lower() and 'false' not in response.lower():
            return True
        elif 'false' in response.lower() and 'true' not in response.lower():
            return False

        print(f"Warning: Unclear response from LLM: {response}")
        return False
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return False

def check_code_quality(instance, min_patch_length=100, max_patch_length=100000):
    """Check basic code quality based on heuristics"""
    patch = instance.get("patch", "")
    
    # Check patch length
    if len(patch) < min_patch_length:
        return False, "Patch too short"
    
    if len(patch) > max_patch_length:
        return False, "Patch too long"
    
    # Check for test files only
    test_files_only = True
    file_pattern = re.compile(r"diff --git a/(.*?) b/")
    file_matches = file_pattern.findall(patch)
    
    if not file_matches:
        return False, "No files found in patch"
    
    for file in file_matches:
        if not (file.endswith("test.py") or file.endswith("tests.py") or 
                "test/" in file or "spec/" in file or file.endswith("_test.go")):
            test_files_only = False
            break
    
    if test_files_only and len(file_matches) > 0:
        return False, "Patch contains only test files"
    
    # Check for meaningful changes (not just comments or whitespace)
    content_lines = 0
    added_lines = re.findall(r'\n\+[^\+]', patch)
    for line in added_lines:
        stripped = line.replace('\n+', '').strip()
        if stripped and not stripped.startswith('//') and not stripped.startswith('#'):
            content_lines += 1
    
    if content_lines < 5:
        return False, "Too few meaningful added lines"
    
    return True, "Passed code quality checks"

def check_problem_quality(instance, disallowed_phrases):
    """Check problem statement quality based on heuristics"""
    problem = instance.get("problem_statement", "")
    
    # Check length
    if len(problem) < 50:
        return False, "Problem statement too short"
    
    if len(problem) > 2000:
        return False, "Problem statement too long"
    
    # Check for disallowed phrases
    lower_problem = problem.lower()
    for phrase in disallowed_phrases:
        if phrase.lower() in lower_problem:
            return False, f"Problem contains banned phrase: {phrase}"
    
    return True, "Passed problem quality checks"

def process_instance(instance):
    glm_api_key = "abvRuB8YBgs92Ns3NzEtaHCkuFV4cJqVVCUJirfe7bqnyeP0mFXKRA4FyaLgctoA"
    glm_base_url = "https://api.chatglm.cn/v1"
    
    disallowed_phrases = [
        "homework", "assignment", "exercise", "I don't know", 
        "I'm not sure", "unclear", "confused", "what do you mean",
        "can you explain", "please clarify", "don't understand"
    ]

    try:
        # Check if required fields exist
        for field in ["problem_statement", "patch"]:
            if field not in instance or not instance[field]:
                return None
        
        # Step 1: Files that have more than 5 diffs
        count = instance["patch"].count("diff --git a/")
        if count > 5:
            return None
        
        # Step 2: Check basic problem quality
        problem_passed, problem_reason = check_problem_quality(instance, disallowed_phrases)
        if not problem_passed:
            return None
        
        # Step 3: Check basic code quality
        code_passed, code_reason = check_code_quality(instance)
        if not code_passed:
            return None
        
        # Step 4: Check if the problem statement is vague (using LLM)
        if is_vague_problem_statement(instance, glm_api_key, glm_base_url):
            return None
        
        # All checks passed, return the instance
        return instance
    except Exception as e:
        print(f"Error processing instance: {str(e)}")
        return None

def process_data_parallel(instances, output_file, num_workers=50):
    results = []
    processed_count = 0
    rejected_count = 0
    
    # Create stats dictionary
    rejection_stats = {
        "total_rejected": 0,
        "reasons": {
            "too_many_diffs": 0,
            "poor_problem_quality": 0,
            "poor_code_quality": 0,
            "vague_problem": 0,
            "processing_error": 0
        }
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_instance, instance)
            for instance in instances
        ]

        with tqdm(total=len(futures), desc="Processing instances") as progress:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        with open(output_file, "a") as f:
                            f.write(json.dumps(result) + "\n")
                        processed_count += 1
                    else:
                        rejected_count += 1
                except Exception as e:
                    print(f"Error in processing: {str(e)}")
                    rejected_count += 1
                    rejection_stats["reasons"]["processing_error"] += 1
                progress.update(1)

    # Calculate acceptance rate
    total = len(instances)
    acceptance_rate = (processed_count / total) * 100 if total > 0 else 0
    
    print(f"\nFiltering Results:")
    print(f"Processed {processed_count} instances successfully out of {total}")
    print(f"Rejected {rejected_count} instances")
    print(f"Acceptance rate: {acceptance_rate:.2f}%")
    
    return results

if __name__ == "__main__":
    input_file = "merged_swe_dataset.jsonl"
    output_file = "processed_swe_dataset.jsonl"

    if os.path.exists(output_file):
        os.remove(output_file)

    with jsonlines.open(input_file, "r") as f:
        instances = list(f)

    process_data_parallel(instances, output_file)