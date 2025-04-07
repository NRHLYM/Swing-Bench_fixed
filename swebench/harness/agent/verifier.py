import os
import copy
from abc import ABC, abstractmethod
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from swebench.harness.constants.swing_constants import(
    AgentState,
    SwingbenchInstance
)
from swebench.harness.router import CIToolBase, CargoCITool
from swebench.harness.router import HANDLER

def create_patch_from_diff(original_code: str, fixed_code: str, file_path: str) -> str:
    """Create a git patch from two versions of code."""
    import tempfile
    import subprocess
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create necessary directories
            file_dir = os.path.dirname(os.path.join(temp_dir, file_path))
            os.makedirs(file_dir, exist_ok=True)
            
            # Initialize git repo
            os.chdir(temp_dir)
            subprocess.run(["git", "init", "-q"], check=True)
            subprocess.run(["git", "config", "user.name", "test"], check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
            
            # Create and add original file
            with open(os.path.join(temp_dir, file_path), "w") as f:
                f.write(original_code)
            
            subprocess.run(["git", "add", file_path], check=True)
            subprocess.run(["git", "commit", "-m", "original", "-q"], check=True)
            
            # Create fixed version
            with open(os.path.join(temp_dir, file_path), "w") as f:
                f.write(fixed_code)
            
            # Get git diff
            result = subprocess.run(
                ["git", "diff", "--no-color"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout if result.stdout else ""
        except subprocess.CalledProcessError as e:
            print(f"Error generating diff: {e}")
            return ""
        except Exception as e:
            print(f"Unexpected error: {e}")
            return ""

def parse_testcase(source_code: str, file_path: str) -> str:
    """Create a git patch for new test file."""
    if not source_code.endswith("\n"):
        source_code += "\n\n"
    elif not source_code.endswith("\n\n"):
        source_code += "\n"
    
    patch = [
        f"diff --git a/{file_path} b/{file_path}",
        "new file mode 100644",
        "index 0000000..0000000",
        f"--- /dev/null",
        f"+++ b/{file_path}",
        "@@ -0,0 +1,{} @@".format(len(source_code.splitlines()))
    ]
    lines = source_code.splitlines()
    formatted_lines = [f"+{line.rstrip()}" for line in lines]
    patch.extend(formatted_lines)
    return "\n".join(patch)

class Verifier:
    @abstractmethod
    def __init__(self, ci_tool: CIToolBase):
        raise NotImplementedError

    @abstractmethod
    def _extract_code(self, input: str):
        raise NotImplementedError

    @abstractmethod
    def verify(self, data: SwingbenchInstance, input: str):
        raise NotImplementedError


class PatchVerifier(Verifier):
    def __init__(self, ci_tool_name: str, workdir: str = "./testbed", output_dir: str = "./logs", src_folder: str = "./repos"):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.output_dir = output_dir
        self.src_folder = src_folder
    
    # TODO(wdxu): default implementation, need users (competitors) pass their implementations.
    def _extract_code(self, input: str):
        import re
        code_block_pattern = r"```[^\n]*\n(.*?)```"
        matches = re.findall(code_block_pattern, input, re.DOTALL)
        if matches:
            input = matches[0]
        return input.strip()

    def verify(self, data: SwingbenchInstance, input: str):
        workdir = f"{self.workdir}/{data.instance_id}"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # TODO(wdxu): create_diff_patch
        # BasePatchCreater
        code = self._extract_code(input)
        os.makedirs(workdir, exist_ok=True)
        patch = create_patch_from_diff(data.code, code, f'{workdir}/tests/test_{timestamp}.rs')
        new_data = copy.copy(data)
        new_data.patch = patch

        # Check if CI tools are available
        if hasattr(data, 'ci_name_list') and data.ci_name_list:
            # Use CI tool if available
            ci_tool = HANDLER.get(self.ci_tool_name)
            if ci_tool:
                config = {
                    "id": data.instance_id,
                    "repo": data.repo,
                    "base_commit": data.base_commit,
                    "merge_commit": data.merge_commit_sha,
                    "patch": patch,
                    "workdir": workdir,
                    "output_dir": self.output_dir,
                    "apply_patch": True,
                    "ci_name_list": data.ci_name_list
                }
                tool = ci_tool(config)
                result = tool.run_ci()
                # For CI tools, we consider it successful if all tests pass
                is_success = all(test.get('stepResult', '') == 'success' for test in result)
                return {
                    "success": is_success,
                    "tool": self.ci_tool_name,
                    "result": result,
                    "patch": patch
                }
        
        # Fall back to Cargo tool if no CI tools available
        cargo_tool = CargoCITool({
            "id": data.instance_id,
            "repo": data.repo,
            "base_commit": data.base_commit,
            "merge_commit": data.merge_commit_sha,
            "patch": patch,
            "workdir": f"{self.workdir}/{data.instance_id}",
            "output_dir": self.output_dir,
            "apply_patch": True,
            "src_folder": self.src_folder
        })
        log_file = f"{self.output_dir}/{data.instance_id}.log"
        result = cargo_tool.run_ci(log_file)
        # For Cargo, we consider it successful if returncode is 0 and all tests pass
        is_success = result.get('returncode', 1) == 0 and not result.get('test_results', {}).get('failed', [])
        return {
            "success": is_success,
            "tool": "cargo",
            "result": result,
            "patch": patch
        }


class TestVerifier(Verifier):
    def __init__(self, ci_tool: CIToolBase, workdir: str = "./testbed", output_dir: str = "./logs"):
        self.ci_tool = ci_tool
        self.workdir = workdir
        self.output_dir = output_dir
        
    def _extract_code(self, input: str):
        import re
        code_block_pattern = r"```[^\n]*\n(.*?)```"
        matches = re.findall(code_block_pattern, input, re.DOTALL)
        if matches:
            input = matches[0]
        return input.strip()

    def verify(self, data: SwingbenchInstance, input: str):
        # Extract the test patch
        test_patch = self._extract_code(input)
        
        # Create a copy of data with the test patch
        new_data = copy.copy(data)
        new_data.test_patch = test_patch
        
        # Apply the golden patch first
        if hasattr(data, 'patch') and data.patch:
            # Use the same verification logic as PatchVerifier
            patch_verifier = PatchVerifier(
                self.ci_tool.__class__.__name__.lower(), 
                self.workdir, 
                self.output_dir
            )
            patch_result = patch_verifier.verify(data, data.patch)
            
            if not patch_result["success"]:
                return {
                    "success": False,
                    "reason": "Golden patch verification failed",
                    "patch_result": patch_result,
                    "test_patch": test_patch
                }
        
        # Now verify the test patch
        if isinstance(self.ci_tool, CargoCITool):
            # For Cargo, we need to run the tests
            result = self.ci_tool.run_ci(f"./logs/{data.instance_id}_test.log")
            is_success = result.get('returncode', 1) == 0 and not result.get('test_results', {}).get('failed', [])
            return {
                "success": is_success,
                "tool": "cargo",
                "result": result,
                "test_patch": test_patch
            }
        else:
            # For CI tools, run the verification
            result = self.ci_tool.run_ci()
            is_success = all(test.get('stepResult', '') == 'success' for test in result)
            return {
                "success": is_success,
                "tool": "ci",
                "result": result,
                "test_patch": test_patch
            }