import os
import copy
from abc import abstractmethod
from datetime import datetime
from swebench.harness.constants.swing_constants import SwingbenchInstance
from swebench.harness.router import CIToolBase, CargoCITool
from swebench.harness.router import EVAL_HANDLER
from swebench.harness.agent.prompt import GENERATE_TEST_SYSTEM_MESSAGE, GENERATE_TEST_TEMPLATE
from swebench.harness.agent.model import AgentProxy
from swebench.harness.agent.utils import parse_testcase, apply_patch, files_to_str
from swebench.harness.agent.editor import RawDataCodeEditor, generate_git_diff

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
                 workdir: str = "./testbed", 
                 output_dir: str = "./logs", 
                 src_folder: str = "./repos",
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None):
        self.ci_tool_name = ci_tool_name
        self.workdir = workdir
        self.output_dir = output_dir
        self.src_folder = src_folder
        self.code_editor = RawDataCodeEditor(
            api_key=api_key,
            base_url=base_url,
            model=model
        )

    def generate(self, data: SwingbenchInstance):
        fixes = []
        for file_path, original_code in data.retrieved_files.items():
            result = self.code_editor.edit_code(
                issue=data.problem_statement,
                original_code=original_code,
                file_path=file_path
            )
            fixes.append(result["code_edits"][0])

        git_diff = ""
        for fix in fixes:
            git_diff += generate_git_diff(fix["file"], fix["code_to_be_modified"], fix["code_edited"])
            git_diff += "\n"
        return git_diff
    
    def verify(self, data: SwingbenchInstance, patch: str):
        os.makedirs(f"{self.workdir}/{data.instance_id}", exist_ok=True)
        apply_patch(patch, f"{self.workdir}/{data.instance_id}", data.base_commit)
        
        
        # TODO[T0]: need to be revised
        if hasattr(data, 'ci_name_list') and data.ci_name_list:
            ci_tool = EVAL_HANDLER.get(self.ci_tool_name)
            if ci_tool:
                config = {
                    "id": data.instance_id,
                    "repo": data.repo,
                    "base_commit": data.base_commit,
                    "merge_commit": data.merge_commit_sha,
                    "patch": data.patch,
                    "workdir": f"{self.workdir}/{data.instance_id}" ,
                    "output_dir": self.output_dir,
                    "apply_patch": True,
                    "ci_name_list": data.ci_name_list
                }
                tool = ci_tool(config)
                result = tool.run_ci()
                is_success = all(test.get('stepResult', '') == 'success' for test in result)
                return {
                    "success": is_success,
                    "tool": self.ci_tool_name,
                    "result": result,
                    "patch": data.patch
                }
        
        else:
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
            "patch": data.patch
        }


class TestVerifier(Verifier): # haoran: We can just write the test code into a file instead of making a patch
    def __init__(self, ci_name: str, workdir: str = "testbed", 
                 output_dir: str = "logs", src_folder: str = "repos", 
                 proxy: AgentProxy = None):
        self.ci_tool = EVAL_HANDLER.get(ci_name)
        self.workdir = workdir
        self.output_dir = output_dir
        self.src_folder = src_folder
        self.proxy = proxy

    def generate(self, data: SwingbenchInstance):
        prompt = [
            {"role": "system", "content": GENERATE_TEST_SYSTEM_MESSAGE},
            {"role": "user", "content": GENERATE_TEST_TEMPLATE.format(
                issue=data.problem_statement,
                retrieved_code=files_to_str(data.retrieved_files),
                patch=data.patch,
                language=data.language
            )}
        ]
        response = self.proxy.generate(prompt, offline=False)
        testcase = parse_testcase(response, data.language)
        return testcase

    def verify(self, data: SwingbenchInstance, testcase: str, test_name: str):
        os.makedirs(f"{self.workdir}/{data.instance_id}", exist_ok=True)
        
        # TODO[T0]: need to be revised
        # TODO(haoran): add more languages
        if isinstance(self.ci_tool, CargoCITool):
            with open(f"{self.workdir}/tests/test_swingbench.rs", "w") as f:
                f.write(testcase)
            apply_patch(data.patch, f"{self.workdir}/{data.instance_id}", data.base_commit)
            result = self.ci_tool.run_ci(f"logs/{data.instance_id}_test.log")
            status = result.get('returncode', 1) == 0 and not result.get('test_results', {}).get('failed', [])
            return {
                "success": status,
                "tool": "cargo",
                "result": result,
                "testcase": testcase
            }
        else:
            result = self.ci_tool.run_ci()
            is_success = all(test.get('stepResult', '') == 'success' for test in result)
            return {
                "success": is_success,
                "tool": "ci",
                "result": result,
                "testcase": testcase
            }


if __name__ == "__main__":
    from swebench.harness.swing_utils import load_swingbench_dataset
    from swebench.harness.agent.retriever import BM25DiskRetriever
    from swebench.harness.agent.editor import RawDataCodeEditor
    from swebench.harness.agent.model import ModelInfo, AgentProxy

    retriever = BM25DiskRetriever(index_dir="/raid/Swing-Bench/tmpdata/indexes")
    dataset_jsonl_path = '/raid/Swing-Bench/tmpdata/dataset.json'
    dataset = load_swingbench_dataset(dataset_jsonl_path)

    base_url = "https://api.x.ai/v1"
    api_key = os.environ["XAI_API_KEY"]
    model = "grok-2-latest"

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
