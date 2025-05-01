import json
from swing_chunker import CodeChunker
import re
from tqdm import tqdm
import os


def extract_added_code_from_patch(patch: str) -> str:
    """
    从diff patch字符串中提取所有新增的代码行（以+开头，且不是diff元信息），返回合并后的代码字符串。
    """
    added_lines = []
    for line in patch.splitlines():
        # 跳过diff元信息和文件头
        if line.startswith('+++') or line.startswith('---') or line.startswith('diff --git') or line.startswith('@@'):
            continue
        # 只保留以+开头的新增代码（但不是+++, 避免文件头）
        if line.startswith('+') and not line.startswith('+++'):
            # 去掉开头的+号
            added_lines.append(line[1:])
    # 合并为一个代码字符串
    return '\n'.join(added_lines)

def find_json_files(base_dir):
    """Find all json files in the directory structure that match all_tasks.json pattern"""
    json_files = {}
    
    # Check each language directory
    language_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for lang in language_dirs:
        # Skip hidden directories
        if lang.startswith('.'):
            continue
            
        lang_path = os.path.join(base_dir, lang)
        
        # Look for all_tasks.json directly in language directory
        if os.path.exists(os.path.join(lang_path, "all_tasks.json")):
            json_files[lang] = os.path.join(lang_path, "all_tasks.json")
            continue
            
        # Look for instances directory
        instances_dir = os.path.join(lang_path, f"{lang}_instances")
        if os.path.exists(instances_dir):
            if os.path.exists(os.path.join(instances_dir, "all_tasks.json")):
                json_files[lang] = os.path.join(instances_dir, "all_tasks.json")
                
    return json_files

def process_language_data(input_path, output_path, language):
    """Process data for a specific language and create index dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize chunker for the specific language
    chunker = CodeChunker(language=language, chunk_type="function")
    
    # Load data
    print(f"Loading data from: {input_path}")
    with open(input_path, "r") as f:
        data = json.load(f)
    
    output = []
    for item in tqdm(data, desc=f"Processing {language}"):
        problem = item.get("problem_statement", "")
        patch = item.get("patch", "")
        
        # Extract added code
        code = extract_added_code_from_patch(patch)
        
        # Chunk the code
        chunks = chunker.chunk(code)
        for chunk in chunks:
            output.append({
                "question": problem,
                "answer": chunk["code"]
            })
    
    print(f"Saving {language} data to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return len(output)

# Base directory for data
base_data_dir = "/home/xiongjing/Swing-Bench/Swingbench-Train/SwingBench_Data"
output_base_dir = "/home/xiongjing/Swing-Bench/Swingbench-Train/index_data"

# Find all json files to process
json_files = find_json_files(base_data_dir)

print(f"Found {len(json_files)} language datasets to process:")
for lang, file_path in json_files.items():
    print(f"  - {lang}: {file_path}")

# Process each language
results = {}
for lang, input_path in json_files.items():
    output_path = os.path.join(output_base_dir, lang, "index_dataset.json")
    try:
        count = process_language_data(input_path, output_path, lang)
        results[lang] = {"status": "Success", "count": count}
    except Exception as e:
        results[lang] = {"status": "Failed", "error": str(e)}

# Print summary
print("\nProcessing Summary:")
for lang, result in results.items():
    status = result["status"]
    if status == "Success":
        print(f"{lang.upper()}: {status} - {result['count']} Q&A pairs generated")
    else:
        print(f"{lang.upper()}: {status} - {result['error']}")

