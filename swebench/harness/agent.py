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
from swebench.inference.make_datasets.swing_search_index import search_instance
from swebench.harness.router import CIToolBase, CargoCITool, DockerCITool, ActCITool
from swebench.harness.router import HANDLER

OPENAI_LIST = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4.5-preview",
               "/home/mnt/wdxu/models/DeepSeek-R1-Distill-Qwen-7B",
               "/home/mnt/wdxu/models/Qwen2.5-Coder-14B-Instruct",
               "/home/mnt/wdxu/models/Qwen2.5-Coder-32B-Instruct",
               "/app/wdxu/models/Qwen2.5-Coder-32B",
               "/app/wdxu/models/DeepSeek-R1-Distill-Qwen-32B",
               "glm-4-flash"]

MODEL_LIMITS = {
    # "claude-instant-1": 100_000,
    # "claude-2": 100_000,
    # "claude-3-opus-20240229": 200_000,
    # "claude-3-sonnet-20240229": 200_000,
    # "claude-3-haiku-20240307": 200_000,
    "gpt-3.5-turbo": 16_385,
    "gpt-4": 8_192,
    "gpt-4o": 128_000,
    "gpt-4.5-preview": 128_000,
    "glm-4-flash": 32_768,
}

# change to a more efficient template
GENERATE_PATCH_SYSTEM_MESSAGE = "You are an AI Senior Full-Stack Engineer specialized in GitHub issue triage and bug fixing." \
                                "You should only generate the fixed code, without any other text or markdown formatting."
GENERATE_PATCH_TEMPLATE = "You are required to fix the code for the specified issue.\n" \
                          "The issue details: {issue}\n" \
                          "The code snippet: {code_snippset}\n" \
                          "Please provide the complete fixed code without any explanations or markdown."

GENERATE_TEST_SYSTEM_MESSAGE = "You are an AI Test Automation Engineer specializing in generating unit tests." \
                                "You should only generate the test code, without any other text or markdown formatting."
GENERATE_TEST_TEMPLATE = "You are required to develop unit tests for the specified code and its fix.\n" \
                          "The issue details: {issue}\n" \
                          "The code snippet: {code_snippset}\n" \
                          "The fixed code: {patch}\n" \
                          "The test case sample: {sample}\n" \
                          "Please provide the complete test code without any explanations or markdown."

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


class Retriever:
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Retriever is not implemented yet.")

    @abstractmethod
    def retrieve(self, instance: SwingbenchInstance):
        raise NotImplementedError("Retrieve is not implemented yet.")


class BM25DiskRetriever(Retriever):
    def __init__(self, index_dir: str, document_encoding_style: str = "file_name_and_contents"):
        self.index_dir = Path(index_dir)
        self.document_encoding_style = document_encoding_style

    def retrieve(self, instance: SwingbenchInstance):
        results = search_instance(
            instance,
            self.index_dir,
            self.document_encoding_style,
            k=1
        )
        # TODO(wdxu): need some reduce strategies
        
        return results


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


class ModelInfo:
    def __init__(self, name: str, base_url: str = None, api_key: str = None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key


class AgentProxy:
    def __init__(self, model_info: ModelInfo):
        """
        Initialize agent proxy.

        Args:
            name (str): agent type
        """
        self.model_info = model_info
        self.score = 0

    def _call_api(self, prompt: str, state: AgentState):
        """
        Route the prompt to different API.

        Args:
            prompt (str): your prompt
            state (AgentState): the type of this prompt
        """
        response = None
        if self.model_info.name in OPENAI_LIST:
            response = self._call_openai(prompt, state)
        else:
            # TODO(wdxu): offline server
            response = self._call_offline()
        return response

    def _call_openai(self, prompt: str, state: AgentState):
        """
        Openai interface.

        Args:
            base_url (str): the base url of the openai server e.g. http://localhost:8000/v1
            prompt (str): your prompt
            state (AgentState): the type of this prompt
        """
        # For local inference, we don't need an API key
        client = OpenAI(
            api_key=self.model_info.api_key,
            base_url=self.model_info.base_url,
        )
        if state == AgentState.PATCH:
            response = client.chat.completions.create(
                model=self.model_info.name,
                # TODO(wdxu): need to designe a message passer.
                messages=[
                    {"role": "system", "content": GENERATE_PATCH_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
            response = client.chat.completions.create(
                model=self.model_info.name,
                messages=[
                    {"role": "system", "content": GENERATE_TEST_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
        return response

    def _call_offline(self):
        raise NotImplementedError("Offline server is not implemented yet.")

    def generate_patch(self, data: SwingbenchInstance, retriever: Retriever = None):
        """
        Patch generater.

        Args:
            data (SwingbenchInstance): a piece of data from dataset
        """
        issue = data.problem_statement + "\n" + data.hints_text
        code_snippset = None
        if retriever is not None:
            # get code_snippset from retriever
            code_snippset = retriever.retrieve(data)
        prompt = GENERATE_PATCH_TEMPLATE.format(issue=issue, code_snippset=code_snippset)

        return self._call_api(prompt, AgentState.PATCH)

    def generate_test(self, data: SwingbenchInstance, retriever: Retriever):
        """
        Test generater.

        Args:
            data (SwingbenchInstance): a piece of data from dataset
        """
        issue = data.problem_statement + "\n" + data.hints_text
        patch = data.patch
        sample = data.test_patch
        code_snippset = None
        if retriever is not None:
            # get code_snippset from retriever
            code_snippset = retriever.retrieve(data)

        prompt = GENERATE_TEST_TEMPLATE.format(issue=issue, code_snippset=code_snippset, patch=patch, sample=sample)

        return self._call_api(prompt, AgentState.TEST)


if __name__ == "__main__":
    DEBUG_VERIFIER = False

    import swing_utils
    dataset_jsonl_path = '/mnt/Data/wdxu/github/Swing-Bench/tmpdata/dataset.json'
    dataset = swing_utils.load_swingbench_dataset(dataset_jsonl_path)
    index_dir = '/mnt/Data/wdxu/github/Swing-Bench/tmpdata/indexes'

    # model_info = ModelInfo(name="/home/mnt/wdxu/models/Qwen2.5-Coder-14B-Instruct", base_url="http://localhost:8000/v1")
    model_info = ModelInfo(name="/app/wdxu/models/DeepSeek-R1-Distill-Qwen-32B", base_url="http://147.8.182.54:10000/v1")
    agent = AgentProxy(model_info)

    retriever = BM25DiskRetriever(index_dir=index_dir)

    if not DEBUG_VERIFIER:
        for swing_instance in dataset:
            response = agent.generate_patch(swing_instance, retriever)
            print('patch response', response.choices[0].message.content)
            response = agent.generate_test(swing_instance, retriever)
            print('test response', response.choices[0].message.content)

        # results = retriever.retrieve(swing_instance)
        # print('retrieved instance id {} results {}'.format(swing_instance.instance_id, results))

    # debug verifier
    else:
        verifier = Verifier()
