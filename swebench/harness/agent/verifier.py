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
        code_snippset = self.retriever.retrieve(data, k=self.retrieve_file_num)
        file_path_list = [hit["docid"] for hit in code_snippset["hits"]]
        code_snippet_list = [hit["contents"] for hit in code_snippset["hits"]]
        response = self.code_editor.edit_code_batch(data.problem_statement,
                                         code_snippet_list,
                                         file_path_list,
                                         retry=self.agent_retry_times)
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        os.makedirs(base_path, exist_ok=True)

        # convert repo path from x/y to x__y
        repo_path = f"{self.src_folder}/{data.repo.replace('/', '__')}"

        if os.path.exists(base_path):
            # remove existing repo
            shutil.rmtree(base_path)

        shutil.copytree(repo_path, base_path)
        subprocess.run(["git", "checkout", data.base_commit], cwd=base_path)

        patch = generate_git_diff_batch(response["code_edits"], base_path)
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        return patch


class TestGenerator(Generator):
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


class PatchVerifier(Verifier):
    def __init__(self, ci_tool_name: str, 
                 workdir: str = "testbed", 
                 src_folder: str = "repos",
                 ):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.src_folder = src_folder

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
                 ):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.src_folder = src_folder
        self.proxy = proxy

    def verify(self, data: SwingbenchInstance, testcase: str):
        # TODO(haoran): add more languages
        base_path = f"{self.workdir}/{data.instance_id}_{str(uuid4())}"
        with open(f"{base_path}/tests/test_swing.rs", "w") as f:
            f.write(testcase)

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
    
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = os.environ["QWEN_API_KEY"]
    model = "qwen-plus"

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
    patch_generator = PatchGenerator(
        workdir=os.environ["SWING_TESTBED_PATH"], 
        src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
        code_editor=code_editor,
        retriever=retriever,
        retrieve_file_num=20,
        agent_retry_times=3
    )
    patch = patch_generator.generate(data)

    print('generated patch: ', patch)

    patch_verifier = PatchVerifier(
        ci_tool_name="cargo", 
        workdir=os.environ["SWING_TESTBED_PATH"], 
        src_folder=os.environ["SWING_REPOS_DIR_PATH"], 
        retrieve_file_num=20,
        agent_retry_times=3
    )
    result = patch_verifier.verify(data, patch)
    print('verify result: ', result)

    # retriever = BM25DiskRetriever(index_dir="/raid/Swing-Bench/tmpdata/indexes")
    # dataset_jsonl_path = '/raid/Swing-Bench/tmpdata/dataset.json'
    # dataset = load_swingbench_dataset(dataset_jsonl_path)



    # code_editor = RawDataCodeEditor(
    #     api_key=api_key,
    #     base_url=base_url,
    #     model=model
    # )

    # for instance in dataset:
    #     code_snippset = retriever.retrieve(instance, k=20)
    #     # print(code_snippset)
    #     file_path_list = [hit["docid"] for hit in code_snippset["hits"]]
    #     code_snippet_list = [hit["contents"] for hit in code_snippset["hits"]]
    #     response = code_editor.edit_code_batch(instance.problem_statement,
    #                                      code_snippet_list,
    #                                      file_path_list,
    #                                      retry=3)
    #     print(response)