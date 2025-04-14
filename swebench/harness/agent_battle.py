import logging
import sys
import os
import platform

from copy import deepcopy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List
from swebench.harness.constants.swing_constants import SwingbenchInstance
from swebench.harness.swing_utils import (
    load_swingbench_dataset,
)

from swebench.harness.agent.model import ModelInfo
from swebench.harness.agent.verifier import PatchVerifier, TestVerifier, PatchGenerator, TestGenerator
from swebench.harness.agent.editor import CodeEditorBase, RawDataCodeEditor
from swebench.harness.agent.retriever import BM25DiskRetriever, Retriever

from swebench.harness.swing_utils import merge_diffs


if platform.system() == "Linux":
    import resource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_battle.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("agent_battle")


def construct_base_instance(data: SwingbenchInstance):
    base_instance = deepcopy.copy(data)
    base_instance.patch = ''
    base_instance.test_patch = ''
    base_instance.merge_commit_sha = base_instance.base_commit

    return base_instance


def check_generated_patch(original_patch_result, golden_patch_result, generated_patch_result):

    # [result format]
    # test_results = {
    #     "ci_1": {
    #         "passed": passed_tests,
    #         "failed": failed_tests,
    #         "ignored": ignored_tests,
    #         "failure_details": {}
    #     }, ...
    # }
    result = {}
    if not (original_patch_result.keys() == \
            golden_patch_result.keys() == \
            generated_patch_result.keys()):
        return None

    for ci_name, ci_result in original_patch_result.items():
        pass
    # before PR(base): results_0
    # after PR(golden patch): results_1
    # after PR(generated patch): results_2

    # results_0 & results_1 & results_2: golden patch CI results F or P
    # F P F -> Failed
    # F P P -> Pass
    # P P F -> Failed

    # CI
    # CI_Job_1 P P F
    # CI_Job_2 P P P
    # CI_Job_3 F P P
    # ...

    return result


def check_generated_test(golden_patch_result, generated_test_result):
    return None


def is_valid_result(result):
    return True


def check_patches(golden_patch_result, patch_with_test_verify_result):
    return True


# TODO(haoran): concurrent execution
def battle_one_turn(
    dataset: List[SwingbenchInstance],
    patch_generator: PatchGenerator,
    test_generator: TestGenerator,
    patch_verifier: PatchVerifier,
    test_verifier: TestVerifier,
    turns: int = 10,
):
    """
    The logic of model battle.

    Args:
        dataset (List[SwingbenchInstance]): a list containing multiple instances of SwingbenchInstance
        patch_generator (PatchGenerator): an instance of PatchGenerator
        test_generator (TestGenerator): an instance of TestGenerator
        patch_verifier (PatchVerifier): an instance of PatchVerifier
        test_verifier (TestVerifier): an instance of TestVerifier
        turns (int): the number of turns in the battle
    """
    patch_agent_score = 0
    test_agent_score = 0

    def get_returncode(result):
        for term in result.keys():
            if 'returncode' in term and term['returncode'] != 0:
                return False
        return True


    for data in dataset:
        # -- Prepare Stage:
        # 0. original patch CI: checkout base_commit  -> apply original (base_commit) patch -> run CI -> results_0.

        # clear all patch information, only need to keep the base_commit
        base_instance = construct_base_instance(data)
        original_patch_result = patch_verifier.verify(base_instance, '') # results_0

        # 1. golden patch CI: checkout base_commit -> apply golden (merged_commit) patch -> run CI -> results_1.
        golden_patch_result = patch_verifier.verify(data, '') # results_1

        for i in range(turns):
            # -- Stage 1: patch, test individually generation and verification.
            
            # Case 1: patch generation and verification.
            patch = patch_generator.generate(data)
            generated_patch_result = patch_verifier.verify(data, patch) # results_2

            # Check if generated patch is valid.
            patch_verify_result = check_generated_patch(original_patch_result,
                                                        golden_patch_result,
                                                        generated_patch_result)

            if not is_valid_result(patch_verify_result):
                patch_agent_score -= 1
                continue

            # Case 2: test generation and verification.
            test = test_generator.generate(data, patch)
            generated_test_result = test_verifier.verify(data, test) # results_3

            # Check if generated test is valid.
            test_verify_result = check_generated_test(golden_patch_result,
                                                      generated_test_result)

            if not is_valid_result(test_verify_result):
                test_agent_score -= 1
                continue

            # -- Stage 2: patch and test generation and verification.

            # Case 3: with new patch, with new generated tests (Verifying)
            patch_with_test = merge_diffs(patch, test)
            patch_with_test_verify_result = test_verifier.verify(data, patch_with_test) # results_4
            
            # Check if patch_with_test is valid.
            patch_with_test_verify_result = check_patches(golden_patch_result,
                                                          patch_with_test_verify_result)

            if not is_valid_result(patch_with_test_verify_result):
                patch_agent_score += 1
            else:
                test_agent_score += 1

    return [patch_agent_score, test_agent_score]            


def battle(
    dataset: List[SwingbenchInstance],
    workdir: str,
    src_folder: str,
    code_editor_lhs: CodeEditorBase,
    code_editor_rhs: CodeEditorBase,
    retriever: Retriever,
    ci_tool_name: str,
    retrieve_file_num: int = 5,
    agent_retry_times: int = 3,
):
    def get_roles(code_editor_lhs, code_editor_rhs):
        patch_verifier = PatchVerifier(ci_tool_name=ci_tool_name, 
            workdir=workdir, 
            src_folder=src_folder, 
        )
        test_verifier = TestVerifier(ci_tool_name=ci_tool_name, 
            workdir=workdir, 
            src_folder=src_folder, 
        )
        patch_generator = PatchGenerator(workdir=workdir, 
            src_folder=src_folder, 
            code_editor=code_editor_lhs,
            retriever=retriever,
            retrieve_file_num=retrieve_file_num,
            agent_retry_times=agent_retry_times
        )
        test_generator = TestGenerator(workdir=workdir, 
            src_folder=src_folder, 
            code_editor=code_editor_rhs,
            retriever=retriever,
            retrieve_file_num=retrieve_file_num,
            agent_retry_times=agent_retry_times
        )
        return patch_generator, test_generator, patch_verifier, test_verifier

    patch_generator, test_generator, patch_verifier, test_verifier = \
        get_roles(code_editor_lhs, code_editor_rhs)
    result = battle_one_turn(dataset,
                             patch_generator,
                             test_generator,
                             patch_verifier,
                             test_verifier)

    patch_generator, test_generator, patch_verifier, test_verifier = \
        get_roles(code_editor_rhs, code_editor_lhs)
    result_rev = battle_one_turn(dataset,
                                 patch_generator,
                                 test_generator,
                                 patch_verifier,
                                 test_verifier)
    
    return result, result_rev


def main(
    dataset_name: str,
    workdir: str,
    src_folder: str,
    open_file_limit: int,
    api_key_lhs: str,
    base_url_lhs: str,
    model_lhs: str,
    api_key_rhs: str,
    base_url_rhs: str,
    model_rhs: str,
    retriever_index_dir: str,
    ci_tool_name: str,
):
    """
    Runs evaluation to battle two agents on a dataset.
    """

    if platform.system() == "Linux":
        logger.info(f"Setting open file limit to {open_file_limit}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    dataset = load_swingbench_dataset(dataset_name)

    retriever = BM25DiskRetriever(index_dir=retriever_index_dir)

    code_editor_lhs = RawDataCodeEditor(
        api_key=api_key_lhs,
        base_url=base_url_lhs,
        model=model_lhs
    )
    code_editor_rhs = RawDataCodeEditor(
        api_key=api_key_rhs,
        base_url=base_url_rhs,
        model=model_rhs
    )

    retrieve_file_num = 5
    agent_retry_times = 3

    result, result_rev = battle(dataset,
                                workdir,
                                src_folder,
                                code_editor_lhs,
                                code_editor_rhs,
                                retriever,
                                ci_tool_name,
                                retrieve_file_num,
                                agent_retry_times)

    print(result)
    print(result_rev)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Runs evaluation harness to compare two agents on a dataset",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    # Common args
    parser.add_argument(
        "--dataset_name",
        default="tmpdata/dataset.json",
        type=str,
        help="Name of dataset or path to JSON file.",
    )

    # default models
    base_url = "http://147.8.181.248:8000/v1/"
    api_key = "no-api-key"
    model = "/home/mnt/wdxu/models/Qwen2.5-Coder-7B-Instruct"

    # Local execution args
    parser.add_argument(
        "--workdir", type=str, default=os.environ["SWING_TESTBED_PATH"], help="Work directory"
    )
    parser.add_argument(
        "--src_folder", type=str, default=os.environ["SWING_REPOS_DIR_PATH"], help="Source code folder"
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )
    parser.add_argument(
        "--api_key_lhs", type=str, default=api_key, help="API key for lhs"
    )
    parser.add_argument(
        "--base_url_lhs", type=str, default=base_url, help="Base URL for lhs"
    )
    parser.add_argument(
        "--model_lhs", type=str, default=model, help="Model for lhs"
    )
    parser.add_argument(
        "--api_key_rhs", type=str, default=api_key, help="API key for rhs"
    )
    parser.add_argument(
        "--base_url_rhs", type=str, default=base_url, help="Base URL for rhs"
    )
    parser.add_argument(
        "--model_rhs", type=str, default=model, help="Model for rhs"
    )
    parser.add_argument(
        "--retriever_index_dir", type=str, default=os.environ["SWING_INDEXES_PATH"], help="Retriever index directory"
    )
    parser.add_argument(
        "--ci_tool_name", type=str, default='cargo', help="CI tool name"
    )
    args = parser.parse_args()
    main(**vars(args))
