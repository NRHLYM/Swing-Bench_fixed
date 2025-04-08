import tempfile
import json
import os
import re
import subprocess

from openai import OpenAI
from abc import ABC, abstractmethod

def remove_line_number(content):
    return re.sub(r"^\d+\s", "", content, flags=re.MULTILINE)

def remove_empty_line(code):
    lines = code.splitlines()
    filtered_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(filtered_lines)

def load_from_repo_structure(file_path, repo_structure, decoding="utf-8"):
    path_parts = file_path.split("/")
    if len(path_parts) == 1:
        path_parts.insert(0, "")
    file_content = repo_structure
    for part in path_parts:
        if part in file_content:
            file_content = file_content[part]
        else:
            return ""
    if isinstance(file_content, dict) and "text" in file_content:
        text_lines = [
            line.encode("ISO-8859-1").decode(decoding) for line in file_content["text"]
        ]
        return "\n".join(text_lines)
    return ""

class CodeEditorBase:
    @abstractmethod
    def __init__(self, api_key: str, base_url: str, model: str):
        raise NotImplementedError

    @abstractmethod
    def edit_code(self, issue: str, original_code: str, file_path: str):
        raise NotImplementedError


class RawDataCodeEditor(CodeEditorBase):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.system_prompt = "Analyze and modify code to resolve issues while preserving functionality. You should use code_editor to process the intput field information. You should use <response>...</response> to wrap the code_editor output."
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        # TODO(haoran): better prompt
        # TODO(haoran): supporting more files
        # TODO(haoran): add naive function calling
        self.function = {
            "name": "code_editor",
            "description": f"{self.system_prompt}",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning_trace": {
                        "type": "string",
                        "description": "Step-by-step analysis of the issue, explanation of the root cause, and justification for the proposed solution"
                    },
                    "code_edits": {
                        "type": "array",
                        "description": "List of specific code modifications required to resolve the issue",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {
                                    "type": "string",
                                    "description": "Relative path to the file that contains code requiring modification"
                                },
                                "code_to_be_modified": {
                                    "type": "string",
                                    "description": "Exact code segment that needs to be changed (must match a portion of the original file)"
                                },
                                "code_edited": {
                                    "type": "string",
                                    "description": "Improved version of the code segment that fixes the issue while maintaining compatibility with surrounding code"
                                }
                            },
                            "required": ["file", "code_to_be_modified", "code_edited"]
                        }
                    }
                },
                "required": ["reasoning_trace", "code_edits"]
            }
        }

    def edit_code(self, issue: str, original_code: str, file_path: str):
        self.function["input"] = {
                "issue": issue,
                "original_code": original_code,
                "file_path": file_path,
        }
        input = json.dumps(self.function)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": input},
                      {"role": "system", "content": self._system_prompt}
                      ],
            temperature=0.0,
            # functions=self.function,
            # function_call={"name": "code_editor"},
        )
        function_call_args = self._parse_structured_data(response.choices[0].message.content)
        if function_call_args is None:
            return None
        return {
            "reasoning_trace": function_call_args["reasoning_trace"],
            "code_edits": function_call_args["code_edits"],
        }

    def edit_code_batch(self, issue: str, original_code: list[dict], file_path_list: list[str]):
        self.function["input"] = {
                "issue": issue,
                "original_code": original_code,
                "file_path_list": file_path_list,
        }
        input = json.dumps(self.function)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": input},
                      {"role": "system", "content": "Analyze and modify code to resolve issues while preserving functionality. You should use code_editor to process the intput field information. You should use <response>...</response> to wrap the code_editor output."}
                      ],
            temperature=0.0,
        )
        function_call_args = self._parse_structured_data(response.choices[0].message.content)
        if function_call_args is None:
            return None
        return {
            "reasoning_trace": function_call_args["reasoning_trace"],
            "code_edits": function_call_args["code_edits"],
        }

    def _parse_structured_data(self, content: str) -> dict:
        pattern = r'<response>\s*(.*?)\s*</response>'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return content
        json_content = match.group(1).strip()
        try:
            json_result = json.loads(json_content)
            return json_result
        except json.JSONDecodeError:
            print(f"Failed to parse structured data: {json_content}")
            return None


# TODO(haoran): use flake8 to lint the code
def lint_code(code: str, prev_code: str = ""):
    """
    Lints Python code using flake8 to check for fatal errors.
    
    Args:
        code: Current code to be linted
        prev_code: Previous version of the code (optional)
    
    Returns:
        Tuple containing:
        - Boolean indicating if the code passed linting (no new errors)
        - Set of errors in the previous code
        - Set of errors in the current code
    """
    # Fatal error codes to check for
    fatal_errors = "E9,F821,F823,F831,F406,F407,F701,F702,F704,F706"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "code_to_lint.py")
        
        # Lint previous code if provided
        prev_errors = set()
        if prev_code:
            with open(temp_file, "w") as f:
                f.write(prev_code)
                
            result = subprocess.run(
                f"flake8 --select={fatal_errors} --isolated {temp_file}",
                shell=True,
                capture_output=True,
            )
            error_output = result.stdout.decode("utf-8")
            
            if error_output:
                for error in error_output.split(f"{temp_file}:")[1:]:
                    # Extract error message without line number
                    error_msg = ":".join(error.split(":")[2:]).strip()
                    prev_errors.add(error_msg)
        
        # Lint current code
        with open(temp_file, "w") as f:
            f.write(code)
            
        result = subprocess.run(
            f"flake8 --select={fatal_errors} --isolated {temp_file}",
            shell=True,
            capture_output=True,
        )
        error_output = result.stdout.decode("utf-8")
        
        current_errors = set()
        if error_output:
            for error in error_output.split(f"{temp_file}:")[1:]:
                error_msg = ":".join(error.split(":")[2:]).strip()
                current_errors.add(error_msg)
        
        # Check for new errors
        new_errors = current_errors - prev_errors
        if new_errors:
            return False, prev_errors, current_errors
            
        return True, set(), set()

def generate_git_diff(file_path: str, old_content: str, new_content: str):
    """
    Creates a temporary git repository and returns the diff between two versions of a file.
    
    Args:
        file_path: Path to the file within the repository
        old_content: Initial content of the file
        new_content: Modified content of the file
    
    Returns:
        String containing the git diff output
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        subprocess.run("git init -b main -q", shell=True, cwd=tmp_dir)
        subprocess.run("git config user.name 'test'", shell=True, cwd=tmp_dir)
        subprocess.run("git config user.email 'test@example.com'", shell=True, cwd=tmp_dir)        
        file_dir = os.path.dirname(file_path)
        if file_dir:
            os.makedirs(os.path.join(tmp_dir, file_dir), exist_ok=True)

        with open(os.path.join(tmp_dir, file_path), "w") as f:
            f.write(old_content)
        subprocess.run(
            f"git add {file_path} && git commit -m 'initial commit'",
            shell=True,
            cwd=tmp_dir
        )
        
        with open(os.path.join(tmp_dir, file_path), "w") as f:
            f.write(new_content)
        result = subprocess.run(
            f"git diff {file_path}", 
            shell=True, 
            capture_output=True,
            cwd=tmp_dir
        )
        diff_output = result.stdout.decode("utf-8")
        
        return diff_output


if __name__ == "__main__":
    from swebench.harness.agent.model import ModelInfo
    code_editor = RawDataCodeEditor(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
        model="grok-2-latest"
    )
    # code_editor = RawDataCodeEditor(
    #     api_key="no-api-key",
    #     base_url="http://localhost:8000/v1",
    #     model="/home/mnt/wdxu/models/Qwen2.5-Coder-7B-Instruct"
    # )
    with open("/mnt/Data/wdxu/github/Swing-Bench/tmpdata/tset_editor.py", "r") as f:
        content = f.read()
        result = code_editor.edit_code(
            issue="fix the bug",
            original_code=content,
            file_path="/mnt/Data/wdxu/github/Swing-Bench/tmpdata/tset_editor.py"
        )
        print(result)
