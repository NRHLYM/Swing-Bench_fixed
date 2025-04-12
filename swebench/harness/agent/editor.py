import tempfile
import json
import os
import re
import subprocess
import copy
from openai import OpenAI
from abc import ABC, abstractmethod
from swebench.harness.agent.prompt import swing_patch_retry_prompt, swing_test_retry_prompt, swing_patch_function, swing_test_function, swing_patch_system_prompt, swing_test_system_prompt

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
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _parse_structured_data(self, content: str) -> dict:
        pattern = r'<response>\s*(.*?)\s*</response>'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return None, content
        json_content = match.group(1).strip()
        try:
            json_result = json.loads(json_content)
            return json_result, ""
        except json.JSONDecodeError:
            return None, json_content

    def _call_api(self, origin_input: str, role: str, retry: int = 1):
        input = origin_input
        function_call_args, raw_resposne = None, ""
        for i in range(retry):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": input},
                        {"role": "system", "content": swing_patch_system_prompt if role == "patch" else swing_test_system_prompt}],
                temperature=0.0,
                max_completion_tokens=32768,
                
            )
            print('response: ', response.choices[0].message.content)
            function_call_args, raw_resposne = self._parse_structured_data(response.choices[0].message.content)
            if function_call_args == None:
                input = origin_input + "\n " + \
                    (swing_patch_retry_prompt if role == "patch" else swing_test_retry_prompt) + raw_resposne
                continue
            else:
                break
        return function_call_args

    def edit_code(self, issue: str, original_code: str, file_path: str, role: str, retry: int = 1, original_patch: str = None):
        self.function = copy.deepcopy(swing_patch_function if role == "patch" else swing_test_function)
        self.function["input"] = {
                "issue": issue,
                "original_code": original_code,
                "file_path": file_path,
        }
        if original_patch != None:
            self.function["input"]["original_patch"] = original_patch

        origin_input = json.dumps(self.function)
        function_call_args = self._call_api(origin_input, role, retry)
        if function_call_args is None:
            return None
        if role == "patch":
            return {
                "reasoning_trace": function_call_args["reasoning_trace"],
                "code_edits": function_call_args["code_edits"],
            }
        else:
            return {
                "reasoning_trace": function_call_args["reasoning_trace"],
                "test_cases": function_call_args["test_cases"],
            }

    def edit_code_batch(self, issue: str, original_code: list[dict], file_path_list: list[str], role: str, retry: int = 1, original_patch: str = None):
        self.function = copy.deepcopy(swing_patch_function if role == "patch" else swing_test_function)
        self.function["input"] = {
                "issue": issue,
                "original_code": original_code,
                "file_path": file_path_list,
        }
        if original_patch != None:
            self.function["input"]["original_patch"] = original_patch

        origin_input = json.dumps(self.function)
        function_call_args = self._call_api(origin_input, role, retry)
        if function_call_args is None:
            return None
        if role == "patch":
            return {
                "reasoning_trace": function_call_args["reasoning_trace"],
                "code_edits": function_call_args["code_edits"],
            }
        else:
            return {
                "reasoning_trace": function_call_args["reasoning_trace"],
                "test_cases": function_call_args["test_cases"],
            }


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

        new_errors = current_errors - prev_errors
        if new_errors:
            return False, prev_errors, current_errors
            
        return True, set(), set()

def normalize_code(code):
    lines = code.split('\n')
    normalized_lines = [line.lstrip() for line in lines]
    return '\n'.join(normalized_lines)

def find_code_match(content, code_to_find):
    normalized_snippet = normalize_code(code_to_find)
    content_lines = content.split('\n')
    
    for i in range(len(content_lines)):
        if normalized_snippet.split('\n')[0] in content_lines[i].lstrip():
            potential_match = []
            for j in range(i, min(i + len(code_to_find.split('\n')), len(content_lines))):
                potential_match.append(content_lines[j])
            
            potential_match_str = '\n'.join(potential_match)
            if normalize_code(potential_match_str) == normalized_snippet:
                indentation = ''
                for char in content_lines[i]:
                    if char in (' ', '\t'):
                        indentation += char
                    else:
                        break
                
                return potential_match_str, indentation
    
    return None, None

def apply_indentation(code, indentation):
    lines = code.split('\n')
    indented_lines = [indentation + line if line.strip() else line for line in lines]
    return '\n'.join(indented_lines)

def process_file_edits(file_path, file_edits, original_content, is_new_file):
    modified_content = original_content

    for edit in file_edits:
        if "code_to_be_modified" in edit:
            code_to_be_modified = edit["code_to_be_modified"]
            code_edited = edit["code_edited"]
        else:
            code_to_be_modified = ""
            code_edited = edit["test_code"]

        if is_new_file:
            return code_edited
        
        if code_to_be_modified and code_to_be_modified in modified_content:
            print(f"Replacing code in {file_path}")
            modified_content = modified_content.replace(code_to_be_modified, code_edited)
            continue
        
        if code_to_be_modified:
            match_text, indentation = find_code_match(modified_content, code_to_be_modified)
            if match_text:
                indented_code = apply_indentation(code_edited, indentation)
                modified_content = modified_content.replace(match_text, indented_code)
                continue
        
        print(f"Could not find the code segment to be modified in {file_path}. Appending the new code.")
        if modified_content and not modified_content.endswith('\n'):
            modified_content += '\n'
        modified_content += code_edited
    
    return modified_content

def generate_git_diff_batch(code_edits, base_path):
    """
    Creates git diffs by applying all edits to original files and generating diffs.
    Handles both existing file modifications and new file creation.
    
    Args:
        code_edits: List of code edit objects containing file, code_to_be_modified, and code_edited
        base_path: Base directory containing the original files. (repo folder)
    
    Returns:
        Dictionary with file paths as keys and their corresponding git diff output as values
    """
    edits_by_file = {}
    for edit in code_edits:
        file_path = edit["file"]
        if file_path not in edits_by_file:
            edits_by_file[file_path] = []
        edits_by_file[file_path].append(edit)
    
    diffs = {}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        subprocess.run("git init -b main -q", shell=True, cwd=tmp_dir)
        subprocess.run("git config user.name 'test'", shell=True, cwd=tmp_dir)
        subprocess.run("git config user.email 'test@example.com'", shell=True, cwd=tmp_dir)

        for file_path, file_edits in edits_by_file.items():
            file_dir = os.path.dirname(file_path)
            if file_dir:
                os.makedirs(os.path.join(tmp_dir, file_dir), exist_ok=True)
                os.makedirs(os.path.join(base_path, file_dir), exist_ok=True)
            
            temp_file_path = os.path.join(tmp_dir, file_path)
            original_file_path_in_base = os.path.join(base_path, file_path)
            
            file_exists = os.path.exists(original_file_path_in_base)
            if not file_exists:
                print(f"Warning: Original file not found at {original_file_path_in_base}. Creating a new file.")
                subprocess.run(f"touch {original_file_path_in_base}", shell=True)
            
            original_content = ""
            if file_exists:
                with open(original_file_path_in_base, 'r') as f:
                    original_content = f.read()
            
            is_new_file = not file_exists
            modified_content = process_file_edits(file_path, file_edits, original_content, is_new_file)
            
            with open(temp_file_path, "w") as f:
                f.write(original_content)
            
            subprocess.run(
                f"git add {file_path} && git commit -m 'original code'",
                shell=True,
                cwd=tmp_dir
            )
            
            with open(temp_file_path, "w") as f:
                f.write(modified_content)
            
            result = subprocess.run(
                f"git diff HEAD {file_path}", 
                shell=True, 
                capture_output=True,
                cwd=tmp_dir
            )
            diff_output = result.stdout.decode("utf-8")
            diffs[file_path] = diff_output
            
            subprocess.run(
                f"git add {file_path} && git commit -m 'edited code'",
                shell=True,
                cwd=tmp_dir
            )

    return diffs

if __name__ == "__main__":
    # from swebench.harness.agent.model import ModelInfo
    # code_editor = RawDataCodeEditor(
    #     api_key=os.environ["XAI_API_KEY"],
    #     base_url="https://api.x.ai/v1",
    #     model="grok-2-latest"
    # )
    # code_editor = RawDataCodeEditor(
    #     api_key="no-api-key",
    #     base_url="http://localhost:8000/v1",
    #     model="/home/mnt/wdxu/models/Qwen2.5-Coder-7B-Instruct"
    # )
    # with open("/mnt/Data/wdxu/github/Swing-Bench/tmpdata/tset_editor.py", "r") as f:
    #     content = f.read()
    #     result = code_editor.edit_code(
    #         issue="fix the bug",
    #         original_code=content,
    #         file_path="/mnt/Data/wdxu/github/Swing-Bench/tmpdata/tset_editor.py"
    #     )
    #     print(result)

    resp = {
        "reasoning_trace": "The issue at hand is to eliminate loading a whole file into a vector when processing TAP files in the `rustzx-core` emulator. Currently, the `Tap::from_asset` function reads the entire file into a `Vec<u8>`, which is not memory-efficient, especially for resource-restricted hosts. To resolve this, we need to modify the `Tap::from_asset` function to use a more memory-efficient approach, such as reading the file in chunks and processing it on demand. This will involve changing the way data is stored and accessed within the `Tap` struct, likely using a custom reader or a file-like object that supports seeking and reading on demand. The goal is to maintain the functionality of the TAP loader while reducing memory usage, which will make the emulator more portable to resource-constrained environments.",
        "code_edits": [
            {
            "file": "rustzx-core/src/zx/tape/tap.rs",
            "code_to_be_modified": "let mut buffer = [0u8; 1024];\nlet mut read_bytes = asset.read(&mut buffer)?;\nwhile read_bytes != 0 {\n    tap.data.extend_from_slice(&buffer[0..read_bytes]);\n    read_bytes = asset.read(&mut buffer)?;\n}",
            "code_edited": "use crate::host::LoadableAsset;\n\nstruct TapReader<R: LoadableAsset> {\n    asset: R,\n    position: usize,\n}\n\nimpl<R: LoadableAsset> TapReader<R> {\n    fn new(asset: R) -> Self {\n        TapReader { asset, position: 0 }\n    }\n\n    fn read(&mut self, buf: &mut [u8]) -> Result<usize, R::Error> {\n        self.asset.seek(SeekFrom::Start(self.position as u64))?;\n        let bytes_read = self.asset.read(buf)?;\n        self.position += bytes_read;\n        Ok(bytes_read)\n    }\n}\n\npub fn from_asset(mut asset: impl LoadableAsset) -> Result<Self> {\n    use crate::utils::make_word;\n\n    let mut tap = Self::default();\n\n    let reader = TapReader::new(asset);\n\n    tap.block_info.clear();\n    let mut p = 0;\n    'blocks: loop {\n        let mut len_bytes = [0u8; 2];\n        reader.read(&mut len_bytes)?;\n        let len = make_word(len_bytes[1], len_bytes[0]) as usize;\n        tap.block_info.push(BlockInfo {\n            length: len,\n            pos: p + 2,\n            end: p + 2 + len - 1,\n        });\n        p += 2 + len;\n        if p >= reader.position {\n            break 'blocks;\n        }\n    }\n    tap.reset_state();\n\n    Ok(tap)\n}"
            },
            {
            "file": "rustzx-core/src/zx/tape/tap.rs",
            "code_to_be_modified": "pub struct Tap {\n    /// state of tape\n    state: TapeState,\n    /// previous state\n    prev_state: TapeState,\n    /// data of tape\n    data: Vec<u8>,\n    /// fields for pulse making from byte\n    curr_bit: bool,\n    curr_byte: u8,\n    curr_mask: u8,\n    // pulses left to next state\n    pulse_counter: usize,\n    /// block info\n    block_info: Vec<BlockInfo>,\n    block: usize,\n    pos_in_block: usize,\n    /// between-state timings\n    delay: Clocks,\n    acc_clocks: Clocks,\n}",
            "code_edited": "use crate::host::LoadableAsset;\n\npub struct Tap<R: LoadableAsset> {\n    /// state of tape\n    state: TapeState,\n    /// previous state\n    prev_state: TapeState,\n    /// reader for tape data\n    reader: TapReader<R>,\n    /// fields for pulse making from byte\n    curr_bit: bool,\n    curr_byte: u8,\n    curr_mask: u8,\n    // pulses left to next state\n    pulse_counter: usize,\n    /// block info\n    block_info: Vec<BlockInfo>,\n    block: usize,\n    pos_in_block: usize,\n    /// between-state timings\n    delay: Clocks,\n    acc_clocks: Clocks,\n}"
            },
            {
            "file": "rustzx-core/src/zx/tape/tap.rs",
            "code_to_be_modified": "fn block_byte(&self, offset: usize) -> Option<u8> {\n    if self.block_info.is_empty() {\n        return None;\n    };\n    let block = self.block_info[self.block];\n    if offset < block.length {\n        Some(self.data[block.pos + offset])\n    } else {\n        None\n    }\n}",
            "code_edited": "fn block_byte(&mut self, offset: usize) -> Option<u8> {\n    if self.block_info.is_empty() {\n        return None;\n    }\n    let block = self.block_info[self.block];\n    if offset < block.length {\n        let mut byte = [0u8; 1];\n        self.reader.seek(SeekFrom::Start((block.pos + offset) as u64)).unwrap();\n        self.reader.read(&mut byte).unwrap();\n        Some(byte[0])\n    } else {\n        None\n    }\n}"
            }
        ]
    }

    test_resp = {'reasoning_trace': 'The code in question is responsible for loading a TAP file into a vector. The current implementation reads the file into a buffer of size 1024 bytes and extends the vector with the contents of the buffer until the end of the file is reached. The proposed changes aim to eliminate the loading of the whole file into a vector by using a loop to read the file in chunks until the end is reached. This change is necessary to allow the code to be ported to more resource-restricted hosts.', 'test_cases': [{'file': 'tests/tap_loader_test.rs', 'test_name': 'Test loading a small TAP file', 'test_code': 'use rustzx_core::zx::tape::tap::Tap;\
    use std::fs::File;\
    use std::io::Read;\
    \
    #[test]\
    fn test_load_small_tap_file() {\
        let mut file = File::open("test_data/small_tap.tap").unwrap();\
        let mut tap = Tap::default();\
        let mut buffer = [0u8; 1024];\
        let mut read_bytes = file.read(&mut buffer).unwrap();\
        while read_bytes != 0 {\
            tap.data.extend_from_slice(&buffer[0..read_bytes]);\
            read_bytes = file.read(&mut buffer).unwrap();\
        }\
        assert_eq!(tap.data.len(), 100); // Assuming the small TAP file has 100 bytes\
        }', 'test_description': 'This test case verifies that the TAP file loader can correctly load a small TAP file into the vector.'}, {'file': 'tests/tap_loader_test.rs', 'test_name': 'Test loading a large TAP file', 'test_code': 'use rustzx_core::zx::tape::tap::Tap;\
    use std::fs::File;\
    use std::io::Read;\
    \
    #[test]\
    fn test_load_large_tap_file() {\
        let mut file = File::open("test_data/large_tap.tap").unwrap();\
        let mut tap = Tap::default();\
        let mut buffer = [0u8; 1024];\
        let mut read_bytes = file.read(&mut buffer).unwrap();\
        while read_bytes != 0 {\
            tap.data.extend_from_slice(&buffer[0..read_bytes]);\
            read_bytes = file.read(&mut buffer).unwrap();\
        }\
        assert_eq!(tap.data.len(), 100000); // Assuming the large TAP file has 100000 bytes\
        }', 'test_description': 'This test case verifies that the TAP file loader can correctly load a large TAP file into the vector.'}, {'file': 'tests/tap_loader_test.rs', 'test_name': 'Test loading an empty TAP file', 'test_code': 'use rustzx_core::zx::tape::tap::Tap;\
    use std::fs::File;\
    use std::io::Read;\
    \
    #[test]\
    fn test_load_empty_tap_file() {\
        let mut file = File::open("test_data/empty_tap.tap").unwrap();\
        let mut tap = Tap::default();\
        let mut buffer = [0u8; 1024];\
        let mut read_bytes = file.read(&mut buffer).unwrap();\
        while read_bytes != 0 {\
            tap.data.extend_from_slice(&buffer[0..read_bytes]);\
            read_bytes = file.read(&mut buffer).unwrap();\
        }\
        assert_eq!(tap.data.len(), 0); // Assuming the empty TAP file has 0 bytes\
        }', 'test_description': 'This test case verifies that the TAP file loader can correctly handle an empty TAP file.'}, {'file': 'tests/tap_loader_test.rs', 'test_name': 'Test loading a TAP file with a single block', 'test_code': 'use rustzx_core::zx::tape::tap::Tap;\
    use std::fs::File;\
    use std::io::Read;\
    \
    #[test]\
    fn test_load_tap_file_with_single_block() {\
        let mut file = File::open("test_data/single_block_tap.tap").unwrap();\
        let mut tap = Tap::default();\
        let mut buffer = [0u8; 1024];\
        let mut read_bytes = file.read(&mut buffer).unwrap();\
        while read_bytes != 0 {\
            tap.data.extend_from_slice(&buffer[0..read_bytes]);\
            read_bytes = file.read(&mut buffer).unwrap();\
        }\
        assert_eq!(tap.data.len(), 10); // Assuming the single block TAP file has 10 bytes\
        }', 'test_description': 'This test case verifies that the TAP file loader can correctly load a TAP file with a single block.'}, {'file': 'tests/tap_loader_test.rs', 'test_name': 'Test loading a TAP file with multiple blocks', 'test_code': 'use rustzx_core::zx::tape::tap::Tap;\
    use std::fs::File;\
    use std::io::Read;\
    \
    #[test]\
    fn test_load_tap_file_with_multiple_blocks() {\
        let mut file = File::open("test_data/multiple_blocks_tap.tap").unwrap();\
        let mut tap = Tap::default();\
        let mut buffer = [0u8; 1024];\
        let mut read_bytes = file.read(&mut buffer).unwrap();\
        while read_bytes != 0 {\
            tap.data.extend_from_slice(&buffer[0..read_bytes]);\
            read_bytes = file.read(&mut buffer).unwrap();\
        }\
        assert_eq!(tap.data.len(), 100); // Assuming the multiple blocks TAP file has 100 bytes\
        }', 'test_description': 'This test case verifies that the TAP file loader can correctly load a TAP file with multiple blocks.'}, {'file': 'tests/tap_loader_test.rs', 'test_name': 'Test loading a TAP file with a single block and a single byte', 'test_code': 'use rustzx_core::zx::tape::tap::Tap;\
    use std::fs::File;\
    use std::io::Read;\
    \
    #[test]\
    fn test_load_tap_file_with_single_block_and_single_byte() {\
        let mut file = File::open("test_data/single_block_single_byte_tap.tap").unwrap();\
        let mut tap = Tap::default();\
        let mut buffer = [0u8; 1024];\
        let mut read_bytes = file.read(&mut buffer).unwrap();\
        while read_bytes != 0 {\
            tap.data.extend_from_slice(&buffer[0..read_bytes]);\
            read_bytes = file.read(&mut buffer).unwrap();\
        }\
        assert_eq!(tap.data.len(), 1); // Assuming the single block single byte TAP file has 1 byte\
        }', 'test_description': 'This test case verifies that the TAP file loader can correctly load a TAP file with a single block and a single byte.'}, {'file': 'tests/tap_loader_test.rs', 'test_name': 'Test loading a TAP file with a single block and multiple bytes', 'test_code': 'use rustzx_core::zx::tape::tap::Tap;\
    use std::fs::File;\
    use std::io::Read;\
    \
    #[test]\
    fn test_load_tap_file_with_single_block_and_multiple_bytes() {\
        let mut file = File::open("test_data/single_block_multiple_bytes_tap.tap").unwrap();\
        let mut tap = Tap::default();\
        let mut buffer = [0u8; 1024];\
        let mut read_bytes = file.read(&mut buffer).unwrap();\
        while read_bytes != 0 {\
            tap.data.extend_from_slice(&buffer[0..read_bytes]);\
            read_bytes = file.read(&mut buffer).unwrap();\
        }\
        assert_eq!(tap.data.len(), 10); // Assuming the single block multiple bytes TAP file has 10 bytes\
        }', 'test_description': 'This test case verifies that the TAP file loader can correctly load a TAP file with a single block and multiple bytes.'}, {'file': 'tests/tap_loader_test.rs', 'test_name': 'Test loading a TAP file with a single block and a single byte and a single block with multiple bytes', 'test_code': 'use rustzx_core::zx::tape::tap::Tap;\
    use std::fs::File;\
    use std::io::Read;\
    \
    #[test]\
    fn test_load_tap_file_with_single_block_and_single_byte_and_single_block_with_multiple_bytes() {\
        let mut file = File::open("test_data/single_block_single_byte_single_block_multiple_bytes_tap.tap").unwrap();\
        let mut tap = Tap::default();\
        let mut buffer = [0u8; 1024];\
        let mut read_bytes = file.read(&mut buffer).unwrap();\
        while read_bytes != 0 {\
            tap.data.extend_from_slice(&buffer[0..read_bytes]);\
            read_bytes = file.read(&mut buffer).unwrap();\
        }\
        assert_eq!(tap.data.len(), 11); // Assuming the single block single byte single block multiple bytes TAP file has 11 bytes\
        }', 'test_description': 'This test case verifies that the TAP file loader can correctly load a TAP file with a single block and a single byte and a single block with multiple bytes.'}, {'file': 'tests/tap_loader_test.rs', 'test_name': 'Test loading a TAP file with a single block and a single byte and a single block with multiple bytes and a single block with a single byte', 'test_code': 'use rustzx_core::zx::tape::tap::Tap;\
    use std::fs::File;\
    use std::io::Read;\
    \
    #[test]\
    fn test_load_tap_file_with_single_block_and_single_byte_and_single_block_with_multiple_bytes_and_single_block_with_single_byte() {\
        let mut file = File::open("test_data/single_block_single_byte_single_block_multiple_bytes_single_block_single_byte_tap.tap").unwrap();\
        let mut tap = Tap::default();\
        let mut buffer = [0u8; 1024];\
        let mut read_bytes = file.read(&mut buffer).unwrap();\
        while read_bytes != 0 {\
            tap.data.extend_from_slice(&buffer[0..read_bytes]);\
            read_bytes = file.read(&mut buffer).unwrap();\
        }\
        assert_eq!(tap.data.len(), 12); // Assuming the single block single byte single block multiple bytes single block single byte TAP file has 12 bytes\
    }', 'test_description': 'This test case verifies that the TAP file loader can correctly load a TAP file with a single block and a single byte and a single block with multiple bytes and a single block with a single byte.'}]}

    diffs = generate_git_diff_batch(test_resp["test_cases"], os.environ["SWING_REPOS_DIR_PATH"] + "/rustzx__rustzx/")
    for file_path, diff in diffs.items():
        print(f"Diff for {file_path}:")
        print(diff)
        print("-" * 80)