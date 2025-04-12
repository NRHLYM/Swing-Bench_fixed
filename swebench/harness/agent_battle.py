import logging
import sys
import os
import platform
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional, List, Dict, Any
from pathlib import Path
from swebench.harness.agent import (
    AgentProxy,
    Verifier,
)
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

    def check_result(result):
        for term in result.keys():
            if term['returncode'] != 0:
                return False
        return True

    for data in dataset:
        for i in range(turns):
            # Stage 1: patch, test individually generation and verification.

            # Generate patch
            patch = patch_generator.generate(data)

            # Verify patch
            patch_verify_result = patch_verifier.verify(data, patch)
            if check_result(patch_verify_result):
                patch_agent_score -= 1
                continue
                
            # Generate test
            test = test_generator.generate(data, patch)

            # Verify test
            test_verify_result = test_verifier.verify(data, test)
            if check_result(test_verify_result):
                test_agent_score -= 1
                continue

            # Stage 2: patch and test generation and verification.
            patch_with_test = merge_diffs(patch, test)
            patch_with_test_verify_result = test_verifier.verify(data, patch_with_test)
            if check_result(patch_with_test_verify_result):
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
    agent = agent.split(",")
    logger.info(f"Processing {dataset_name} for agent {agent[0]} and agent {agent[1]}")

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
        "--api_key_lhs", type=str, default=os.environ["SWING_CODE_EDITOR_API_KEY"], help="API key for lhs"
    )
    parser.add_argument(
        "--base_url_lhs", type=str, default=os.environ["SWING_CODE_EDITOR_BASE_URL"], help="Base URL for lhs"
    )
    parser.add_argument(
        "--model_lhs", type=str, default=os.environ["SWING_CODE_EDITOR_MODEL"], help="Model for lhs"
    )
    parser.add_argument(
        "--api_key_rhs", type=str, default=os.environ["SWING_CODE_EDITOR_API_KEY"], help="API key for rhs"
    )
    parser.add_argument(
        "--base_url_rhs", type=str, default=os.environ["SWING_CODE_EDITOR_BASE_URL"], help="Base URL for rhs"
    )
    parser.add_argument(
        "--model_rhs", type=str, default=os.environ["SWING_CODE_EDITOR_MODEL"], help="Model for rhs"
    )
    parser.add_argument(
        "--retriever_index_dir", type=str, default=os.environ["SWING_INDEXES_PATH"], help="Retriever index directory"
    )
    parser.add_argument(
        "--ci_tool_name", type=str, default='cargo', help="CI tool name"
    )
    args = parser.parse_args()
    main(**vars(args))
