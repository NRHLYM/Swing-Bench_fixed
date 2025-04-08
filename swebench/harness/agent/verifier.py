import os
import copy
from abc import abstractmethod
from datetime import datetime
import subprocess
from uuid import uuid4
from swebench.harness.constants.swing_constants import SwingbenchInstance
from swebench.harness.router import CIToolBase, CargoCITool
from swebench.harness.router import EVAL_HANDLER
from swebench.harness.agent.prompt import GENERATE_TEST_SYSTEM_MESSAGE, GENERATE_TEST_TEMPLATE, TESTCASE_SAMPLE
from swebench.harness.agent.model import AgentProxy
import shutil
from swebench.harness.agent.utils import parse_testcase, apply_git_patch, files_to_str
from swebench.harness.agent.editor import generate_git_diff_batch
from swebench.harness.agent.retriever import Retriever
from swebench.harness.agent.editor import CodeEditorBase

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
    def __init__(self, ci_tool_name: str, 
                 workdir: str = "testbed", 
                 src_folder: str = "repos",
                 code_editor: CodeEditorBase = None,
                 retriever: Retriever = None,
                 retrieve_times: int = 20,
                 retry_times: int = 3,
                 ):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.src_folder = src_folder
        self.retriever = retriever
        self.code_editor = code_editor
        self.retrieve_times = retrieve_times
        self.retry_times = retry_times

    def generate(self, data: SwingbenchInstance):
        """
        Generate a patch for the given Swingbench instance.
        """
        code_snippset = self.retriever.retrieve(data, k=self.retrieve_times)
        file_path_list = [hit["docid"] for hit in code_snippset["hits"]]
        code_snippet_list = [hit["contents"] for hit in code_snippset["hits"]]
        response = self.code_editor.edit_code_batch(data.problem_statement,
                                         code_snippet_list,
                                         file_path_list,
                                         retry=self.retry_times)
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        os.makedirs(base_path, exist_ok=True)
        repo_path = f"{self.src_folder}/{data.repo}"
        shutil.copytree(repo_path, base_path)
        subprocess.run(["git", "checkout", data.base_commit], cwd=base_path)
        patch = generate_git_diff_batch(response, base_path)
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        return patch

    def verify(self, data: SwingbenchInstance, patch: str):
        data.patch = patch
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        
        # CITool will handle the patch
        config = {
            "id": data.instance_id,
            "repo": data.repo,
            "base_commit": data.base_commit,
            "merge_commit": data.merge_commit_sha,
            "patch": patch,
            "src_folder": self.src_folder,
            "output_dir": "logs",
            "workdir": base_path,
            "apply_patch": True,
            "ci_name_list": data.ci_name_list
        }
        ci_tool = EVAL_HANDLER.get(self.ci_tool_name)
        tool = ci_tool(config)
        result = tool.run_ci()
        # TODO(wdxu): unify the return format of run_ci
        test_results = result.get('test_results', {})
        # haoran: FOR CARGO
        # test_results = {
        #     "passed": passed_tests,
        #     "failed": failed_tests,
        #     "ignored": ignored_tests,
        #     "failure_details": {}
        # }

        # TODO: parse success for different ci tools
        def parse_success(test_results):
            pass
        success = parse_success(test_results)
        return {
            "success": success,
            "tool": self.ci_tool_name,
            "result": result,
            "patch": patch
        }


class TestVerifier(Verifier):
    def __init__(self, ci_tool_name: str, 
                 workdir: str = "testbed", 
                 src_folder: str = "repos",
                 proxy: AgentProxy = None,
                 retriever: Retriever = None,
                 retrieve_times: int = 20,
                 retry_times: int = 3,
                 ):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.src_folder = src_folder
        self.proxy = proxy
        self.retriever = retriever
        self.retrieve_times = retrieve_times
        self.retry_times = retry_times

    def generate(self, data: SwingbenchInstance):
        prompt = [
            {"role": "system", "content": GENERATE_TEST_SYSTEM_MESSAGE},
            {"role": "user", "content": GENERATE_TEST_TEMPLATE.format(
                issue=data.problem_statement,
                code_snippset=files_to_str(data.retrieved_files),
                patch=data.patch,
                sample=TESTCASE_SAMPLE
            )}
        ]
        response = self.proxy.generate(prompt, offline=False)
        testcase = parse_testcase(response, data.language)
        return testcase

    def verify(self, data: SwingbenchInstance, testcase: str):
        # TODO(haoran): add more languages
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        with open(f"{base_path}/tests/test_swing.rs", "w") as f:
            f.write(testcase)

        config = {
            "id": data.instance_id,
            "repo": data.repo,
            "base_commit": data.base_commit,
            "merge_commit": data.merge_commit_sha,
            "patch": patch,
            "src_folder": self.src_folder,
            "output_dir": "logs",
            "workdir": base_path,
            "apply_patch": True,
            "ci_name_list": data.ci_name_list
        }
        ci_tool = EVAL_HANDLER.get(self.ci_tool_name)
        tool = ci_tool(config)
        result = tool.run_ci()
        # TODO(wdxu): unify the return format of run_ci
        test_results = result.get('test_results', {})
        # haoran: FOR CARGO
        # test_results = {
        #     "passed": passed_tests,
        #     "failed": failed_tests,
        #     "ignored": ignored_tests,
        # }
        def parse_success(test_results):
            pass
        success = parse_success(test_results)
        return {
            "success": success,
            "tool": self.ci_tool_name,
            "result": result,
            "testcase": testcase
        }

if __name__ == "__main__":
    from swebench.harness.swing_utils import load_swingbench_dataset
    from swebench.harness.agent.retriever import BM25DiskRetriever
    from swebench.harness.agent.editor import RawDataCodeEditor
    from swebench.harness.agent.model import AgentProxy
    import json
    
    base_url = "https://api.x.ai/v1"
    api_key = os.environ["XAI_API_KEY"]
    model = "grok-2-latest"   
    
    with open("tmpdata/demo_dataset.json", "r") as f:
        dataset = json.load(f)
    with open("tmpdata/demo.patch", "r") as f:
        patch = f.read()

    code_editor = RawDataCodeEditor(
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    patch_verifier = PatchVerifier(
        ci_tool_name="cargo", 
        workdir="testbed", 
        src_folder="/raid/rust-repos/", 
        code_editor=code_editor
    )
    data = SwingbenchInstance(**dataset[0])
    patch_verifier.verify(data, patch)
    exit()
    retriever = BM25DiskRetriever(index_dir="/raid/Swing-Bench/tmpdata/indexes")
    dataset_jsonl_path = '/raid/Swing-Bench/tmpdata/dataset.json'
    dataset = load_swingbench_dataset(dataset_jsonl_path)



    code_editor = RawDataCodeEditor(
        api_key=api_key,
        base_url=base_url,
        model=model
    )

    for instance in dataset:
        code_snippset = retriever.retrieve(instance, k=20)
        # print(code_snippset)
        file_path_list = [hit["docid"] for hit in code_snippset["hits"]]
        code_snippet_list = [hit["contents"] for hit in code_snippset["hits"]]
        response = code_editor.edit_code_batch(instance.problem_statement,
                                         code_snippet_list,
                                         file_path_list,
                                         retry=3)
        print(response)