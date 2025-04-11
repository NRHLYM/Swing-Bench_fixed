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
from swebench.harness.utils import get_available_port_pool

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


class Generator:
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self, data: SwingbenchInstance):
        raise NotImplementedError


class PatchGenerator(Generator):
    def __init__(self, workdir: str = "testbed", 
                 src_folder: str = "repos",
                 code_editor: CodeEditorBase = None,
                 retriever: Retriever = None,
                 retrieve_file_num: int = 20,
                 agent_retry_times: int = 3,
                 ):
        self.workdir = workdir
        self.src_folder = src_folder
        self.code_editor = code_editor
        self.retriever = retriever
        self.retrieve_file_num = retrieve_file_num
        self.agent_retry_times = agent_retry_times

    def generate(self, data: SwingbenchInstance):
        """
        Generate a patch for the given Swingbench instance.
        """
        code_snippet = self.retriever.retrieve(data, k=self.retrieve_file_num)
        file_path_list = [hit["docid"] for hit in code_snippet["hits"]]
        code_snippet_list = [hit["contents"] for hit in code_snippet["hits"]]
        response = self.code_editor.edit_code_batch(data.problem_statement,
                                         code_snippet_list,
                                         file_path_list,
                                         role="patch",
                                         retry=self.agent_retry_times)
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        print('patch generator: creating base path: ', base_path)

        # convert repo path from x/y to x__y
        repo_path = f"{self.src_folder}/{data.repo.replace('/', '__')}"

        if os.path.exists(base_path):
            # remove existing repo
            shutil.rmtree(base_path)

        shutil.copytree(repo_path, base_path)
        subprocess.run(["git", "checkout", data.base_commit], cwd=base_path)

        patch = generate_git_diff_batch(response["code_edits"], base_path)
        # if os.path.exists(base_path):
        #     shutil.rmtree(base_path)
        return patch


class TestGenerator(Generator):
    def __init__(self, workdir: str = "testbed", 
                 src_folder: str = "repos",
                 code_editor: CodeEditorBase = None,
                 retriever: Retriever = None,
                 retrieve_file_num: int = 20,
                 agent_retry_times: int = 3,
                 original_patch: str = None,
                 ):
        self.workdir = workdir
        self.src_folder = src_folder
        self.code_editor = code_editor
        self.retriever = retriever
        self.retrieve_file_num = retrieve_file_num
        self.agent_retry_times = agent_retry_times
        self.original_patch = original_patch

    def generate(self, data: SwingbenchInstance):
        print('test generator: data: ', data)
        # TODO(wdxu): remove this hack.
        data.hints_text += "test, testcase, unittest."
        code_snippet = self.retriever.retrieve(data, k=self.retrieve_file_num)
        file_path_list = [hit["docid"] for hit in code_snippet["hits"]]
        code_snippet_list = [hit["contents"] for hit in code_snippet["hits"]]
        response = self.code_editor.edit_code_batch(data.problem_statement,
                                         code_snippet_list,
                                         file_path_list,
                                         role="test",
                                         retry=self.agent_retry_times,
                                         original_patch=self.original_patch)
        print('test generator: response: ', response)
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        print('test generator: creating base path: ', base_path)

        # convert repo path from x/y to x__y
        repo_path = f"{self.src_folder}/{data.repo.replace('/', '__')}"

        if os.path.exists(base_path):
            # remove existing repo
            shutil.rmtree(base_path)

        shutil.copytree(repo_path, base_path)
        subprocess.run(["git", "checkout", data.base_commit], cwd=base_path)

        patch = generate_git_diff_batch(response["test_cases"], base_path)
        # if os.path.exists(base_path):
        #     shutil.rmtree(base_path)
        return patch


class PatchVerifier(Verifier):
    def __init__(self, ci_tool_name: str, 
                 workdir: str = "testbed", 
                 src_folder: str = "repos",
                 ):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.src_folder = src_folder
        self.port_pool_size = 100

    def verify(self, data: SwingbenchInstance, patch: str):
        data.patch = patch
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        
        # CITool will handle the patch
        config = {
            "instance_id": data.instance_id,
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

        # TODO(wdxu): remove the switch process for run_ci.
        if self.ci_tool_name == "act":
            pool = get_available_port_pool(self.port_pool_size)
            result = tool.run_ci(pool)
        else:
            result = tool.run_ci()

        # haoran: FOR CARGO
        # test_results = {
        #     "unit_test": {
        #         "passed": passed_tests,
        #         "failed": failed_tests,
        #         "ignored": ignored_tests,
        #         "failure_details": {}
        #     }
        # }
        # FOR ACT
        # test_results = {
        #     "ci_1": {
        #         "passed": passed_tests,
        #         "failed": failed_tests,
        #         "ignored": ignored_tests,
        #         "failure_details": {}
        #     }, ...
        # }

        return {
            "tool": self.ci_tool_name,
            "result": result,
            "patch": patch
        }


class TestVerifier(Verifier):
    def __init__(self, ci_tool_name: str, 
                 workdir: str = "testbed", 
                 src_folder: str = "repos",
                 proxy: AgentProxy = None,
                 ):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.src_folder = src_folder
        self.proxy = proxy
        self.port_pool_size = 100

    def verify(self, data: SwingbenchInstance, testcase: str):
        # apply both test patch and original patch
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        
        config = {
            "instance_id": data.instance_id,
            "repo": data.repo,
            "base_commit": data.base_commit,
            "merge_commit": data.merge_commit_sha,
            "patch": testcase,
            "src_folder": self.src_folder,
            "output_dir": "logs",
            "workdir": base_path,
            "apply_patch": True,
            "ci_name_list": data.ci_name_list
        }
        ci_tool = EVAL_HANDLER.get(self.ci_tool_name)
        tool = ci_tool(config)

        # TODO(wdxu): remove the switch process for run_ci.
        if self.ci_tool_name == "act":
            pool = get_available_port_pool(self.port_pool_size)
            result = tool.run_ci(pool)
        else:
            result = tool.run_ci()

        # haoran: FOR CARGO
        # test_results = {
        #     "unit_test": {
        #         "passed": passed_tests,
        #         "failed": failed_tests,
        #         "ignored": ignored_tests,
        #         "failure_details": {}
        #     }
        # }
        # FOR ACT
        # test_results = {
        #     "ci_1": {
        #         "passed": passed_tests,
        #         "failed": failed_tests,
        #         "ignored": ignored_tests,
        #         "failure_details": {}
        #     }, ...
        # }

        return {
            "tool": self.ci_tool_name,
            "result": result,
            "test_cases": data.patch
        }


if __name__ == "__main__":
    from swebench.harness.swing_utils import load_swingbench_dataset
    from swebench.harness.agent.retriever import BM25DiskRetriever
    from swebench.harness.agent.editor import RawDataCodeEditor
    from swebench.harness.agent.model import AgentProxy
    import json
    
    SWING_DEBUG_GENERATE_DRYRUN = False
    
    # base_url = "https://api.x.ai/v1/"
    # api_key = os.environ["XAI_API_KEY"]
    # model = "grok-2-latest"

    base_url = "http://147.8.181.248:8000/v1/"
    api_key = "no-api-key"
    model = "/home/mnt/wdxu/models/Qwen2.5-Coder-7B-Instruct"

    with open(os.environ["SWING_DEMO_DATASET_PATH"], "r") as f:
        dataset = json.load(f)
    with open(os.environ["SWING_DEMO_PATCH_PATH"], "r") as f:
        patch = f.read()

    retriever = BM25DiskRetriever(index_dir=os.environ["SWING_INDEXES_PATH"])

    code_editor = RawDataCodeEditor(
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    data = SwingbenchInstance(**dataset[0])
    if not SWING_DEBUG_GENERATE_DRYRUN:
        print('----------- [BEGIN PATCH GENERATOR] -----------',)
        print('input data: ', data)
        patch_generator = PatchGenerator(workdir=os.environ["SWING_TESTBED_PATH"], 
            src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
            code_editor=code_editor,
            retriever=retriever,
            retrieve_file_num=5,
            agent_retry_times=3
        )
        patch = patch_generator.generate(data)
        print('generated patch: ', patch)
        print('----------- [END PATCH GENERATOR] -----------',)

        print('----------- [BEGIN PATCH VERIFIER] -----------')

        patch_verifier = PatchVerifier(ci_tool_name="cargo", 
            workdir=os.environ["SWING_TESTBED_PATH"], 
            src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
        )
        result = patch_verifier.verify(data, patch)
        print('verify result: ', result)
        print('----------- [END PATCH VERIFIER] -----------')

        print('----------- [BEGIN TEST GENERATOR] -----------')
        test_generator = TestGenerator(workdir=os.environ["SWING_TESTBED_PATH"], 
            src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
            code_editor=code_editor,
            retriever=retriever,
            retrieve_file_num=5,
            agent_retry_times=3,
            original_patch=patch
        )
        testcase = test_generator.generate(data)
        print('generated testcase: ', testcase)
        print('----------- [END TEST GENERATOR] -----------')

        print('----------- [BEGIN TEST VERIFIER] -----------')
        test_verifier = TestVerifier(ci_tool_name="cargo", 
            workdir=os.environ["SWING_TESTBED_PATH"], 
            src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
        )
        result = test_verifier.verify(data, testcase)
        print('test verify result: ', result)
        print('----------- [END TEST VERIFIER] -----------')

        # TODO(wdxu): merge patch and testcase to get final results.

    else:
        import swebench.harness.agent.verifier_test_patch as test_patch
        patch = test_patch.patch
