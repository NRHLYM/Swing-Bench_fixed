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
    agent_one: AgentProxy,
    agent_two: AgentProxy,
    verifier: Verifier,
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

    temperatures = [1 - (i / turns) for i in range(turns)] # fixed temperature: 1 -> 0
    for data in dataset:
        patch_agent, test_agent = agent_one, agent_two
        for i in range(turns):
            patch = patch_agent.generate_patch(data, temperature=temperatures[i])
            if not verifier.verify_patch(data, patch): # patch failed
                patch_agent.score -= 1
                patch_agent, test_agent = test_agent, patch_agent
                continue

            test = test_agent.generate_test(data, temperature=temperatures[i])
            if not verifier.verify_test(data, test): # test failed
                test_agent.score -= 1
                patch_agent, test_agent = test_agent, patch_agent
                continue

            if verifier.verify_patch(data, patch, test): # patch and test both passed
                patch_agent.score += 1
            else:
                test_agent.score += 1

            patch_agent, test_agent = test_agent, patch_agent

    return [agent_one.score, agent_two.score]

def main(
    agent: str,
    dataset_name: str,
    target_dir: str,
    report_dir: str,
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

    # Create target directory if it doesn't exist
    expanded_path = os.path.expanduser(target_dir)
    Path(expanded_path).mkdir(parents=True, exist_ok=True)
    
    # Create base report directory if it doesn't exist
    expanded_path = os.path.expanduser(report_dir)
    Path(expanded_path).mkdir(parents=True, exist_ok=True)

    if platform.system() == "Linux":
        logger.info(f"Setting open file limit to {open_file_limit}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    dataset = load_swingbench_dataset(dataset_name)

    agent_one = AgentProxy(agent[0])
    agent_two = AgentProxy(agent[1])

    result = battle(dataset, agent_one, agent_two)

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
    # parser.add_argument(
    #     "--timeout",
    #     type=int,
    #     default=600,
    #     help="Timeout (in seconds) for running tests for each instance",
    # )
    parser.add_argument(
        "--report_dir", type=str, default="./report", help="Directory to write reports to"
    )
    parser.add_argument(
        "--target_dir", type=str, default="./testbed", help="Directory to clone repo to"
    )
    # parser.add_argument(
    #     "--src_folder", 
    #     type=str, 
    #     default="/raid/rust-repos", 
    #     help="Source folder containing Rust repositories"
    # )
    # parser.add_argument(
    #     "--ci_tool",
    #     type=str,
    #     choices=["act", "cargo"],
    #     default="act",
    #     help="CI tool to use for testing (act or cargo)"
    # )
    # parser.add_argument(
    #     "--concurrent_workers",
    #     type=int,
    #     default=1,
    #     help="Number of concurrent workers (1 means sequential execution)"
    # )
    # parser.add_argument(
    #     "--start_index",
    #     type=int,
    #     default=0,
    #     help="Index of first instance to process"
    # )
    # parser.add_argument(
    #     "--max_instances",
    #     type=int,
    #     default=None,
    #     help="Maximum number of instances to process"
    # )

    args = parser.parse_args()
    main(**vars(args))
