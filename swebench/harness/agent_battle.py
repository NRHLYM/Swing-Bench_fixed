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
from swebench.harness.agent.code_editor import RawDataCodeEditor
from swebench.harness.agent.retriever import BM25DiskRetriever


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
def battle(
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
        agent_one (AgentProxy): an instance of AgentProxy
        agent_two (AgentProxy): an instance of AgentProxy
        turns (int): the number of turns in the battle
    """
    patch_agent_score = 0
    test_agent_score = 0

    def check_result(result):
        for term in result.keys():
            if term['returncode'] != 0:
                return False
        return True

    temperatures = [1 - (i / turns) for i in range(turns)] # fixed temperature: 1 -> 0
    for data in dataset:
        for i in range(turns):
            # Stage 1: patch, test individually generation and verification.
            failed_patch = False
            failed_test = False

            # Generate patch
            patch = patch_generator.generate(data)

            # Verify patch
            patch_verify_result = patch_verifier.verify(data, patch)
            if check_result(patch_verify_result):
                failed_patch = True
                
            # Generate test
            test = test_generator.generate(data)

            # Verify test
            test_verify_result = test_verifier.verify(data, test)
            if check_result(test_verify_result):
                failed_test = True

            # Update score
            if failed_patch:
                patch_agent_score -= 1
            if failed_test:
                test_agent_score -= 1

            if failed_patch or failed_test:
                continue

            # Stage 2: patch and test generation and verification.

    return None

def main(
    agent: str,
    dataset_name: str,
    workdir: str,
    src_folder: str,
    open_file_limit: int,
):
    """
    Runs evaluation to battle two agents on a dataset.

    Args:
        agent (str): agent type
        dataset_name (str): Huggingface dataset name or offline dataset path
        target_dir (str): the repository storage path when ACT is running
        report_dir (str): report saving path
        open_file_limit (int): the upper limit on the number of file descriptors that the current process can open.
    """
    import pdb
    pdb.set_trace()
    agent = agent.split(",")
    logger.info(f"Processing {dataset_name} for agent {agent[0]} and agent {agent[1]}")

    if platform.system() == "Linux":
        logger.info(f"Setting open file limit to {open_file_limit}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    dataset = load_swingbench_dataset(dataset_name)

    retriever = BM25DiskRetriever(index_dir=os.environ["SWING_INDEXES_PATH"])

    code_editor = RawDataCodeEditor(
        api_key=os.environ["SWING_CODE_EDITOR_API_KEY"],
        base_url=os.environ["SWING_CODE_EDITOR_BASE_URL"],
        model=os.environ["SWING_CODE_EDITOR_MODEL"]
    )

    patch_verifier = PatchVerifier(ci_tool_name="cargo", 
        workdir=workdir, 
        src_folder=src_folder, 
    )
    test_verifier = TestVerifier(ci_tool_name="cargo", 
        workdir=workdir, 
        src_folder=src_folder, 
    )
    patch_generator = PatchGenerator(workdir=workdir, 
        src_folder=src_folder, 
        code_editor=code_editor,
        retriever=retriever,
        retrieve_file_num=5,
        agent_retry_times=3
    )
    test_generator = TestGenerator(workdir=workdir, 
        src_folder=src_folder, 
        code_editor=code_editor,
        retriever=retriever,
        retrieve_file_num=5,
        agent_retry_times=3
    )

    result = battle(dataset, patch_generator, test_generator, patch_verifier, test_verifier)
    
    # Switching roles
    patch_generator, test_generator = test_generator, patch_generator

    result = battle(dataset, patch_generator, test_generator, patch_verifier, test_verifier)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Runs evaluation harness to compare two agents on a dataset",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        help="Which two agents you want to battle",
        required=True,
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
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )
    parser.add_argument(
        "--report_dir", type=str, default="./report", help="Directory to write reports to"
    )
    parser.add_argument(
        "--target_dir", type=str, default="./testbed", help="Directory to clone repo to"
    )
    args = parser.parse_args()
    main(**vars(args))
