import json
from swing_chunker import CodeChunker
import re
from tqdm import tqdm

input_path = "/home/xiongjing/Swing-Bench/RGER/SwingBench_Data/rust.json"
output_path = "/home/xiongjing/Swing-Bench/RGER/index_data/rust/index_dataset.json"

chunker = CodeChunker(language="rust", chunk_type="function")

with open(input_path, "r") as f:
    data = json.load(f)

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

output = []
for item in tqdm(data):
    problem = item.get("problem_statement", "")
    patch = item.get("patch", "")
    # 只提取新增的Rust代码部分（可用正则或diff解析）
    code = extract_added_code_from_patch(patch)
    #print("code: ", code)
    #assert 1==0
    # 分块
    chunks = chunker.chunk(code)
    for chunk in chunks:
        output.append({
            "question": problem,
            "answer": chunk["code"]
        })

print("output_path: ", output_path)
with open(output_path, "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

