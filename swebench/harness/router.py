import subprocess
import re
import os
import tempfile
import threading
import json
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

def run_script(script_content, cwd=None):
    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".sh") as temp_script:
        temp_script.write(script_content)
        temp_script.flush()
        temp_path = temp_script.name
        try:
            subprocess.run(["bash", temp_path], 
                           cwd=cwd,
                           check=True, 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        except Exception as e:
            return e
    return None

@dataclass
class Task:
    instance_id: str
    env_script: list[str]
    eval_script: list[str]
    target_dir: str
    output_dir: str
    patch: str = None
    apply_patch: bool = False

class CIToolBase:
    def __init__(self, config):
        self.config = config

    def construct(self):
        pass
    
    def run_ci(self, log_file: str = None):
        pass

class CargoCITool(CIToolBase):
    def __init__(self, config):
        super().__init__(config)
        self.construct()

    def _build_repo_base_env(self):
        script = ["#!/bin/bash"]
        
        repo_dir_name = self.config["repo"].replace('/', '__')
        instance_id = self.config.get("instance_id", "unknown")
        src_path = os.path.join(self.config["src_folder"], repo_dir_name)
        dst_path = os.path.join(self.config["workdir"], f"{self.config['repo'].split('/')[1]}_{instance_id}")
        
        script.append(f"mkdir -p {dst_path}")
        script.append(f"cp -r {src_path}/. {dst_path}/")

        return script

    def _build_eval_script(self):
        instance_id = self.config.get("instance_id", "unknown")
        target_dir = os.path.join(self.config["workdir"], f"{self.config['repo'].split('/')[1]}_{instance_id}")

        script = ["#!/bin/bash", 
                  f"cd {target_dir}",
                 ]

        script.append("git stash -u || true")
        
        if "merge_commit" in self.config and self.config["merge_commit"]:
            script.append("git checkout " + self.config["merge_commit"])
            
            # Apply test_patch if it exists
            if self.config.get("test_patch"):
                test_patch_file = f"{target_dir}/test_patch.diff"
                script.append(f"cat > {test_patch_file} << 'EOL'\n{self.config['test_patch']}\nEOL")
                script.append(f"git apply {test_patch_file} || echo 'Failed to apply test_patch'")
            
            # Apply patch only if apply_patch is specified
            if self.config.get("apply_patch", False) and self.config.get("patch"):
                patch_file = f"{target_dir}/patch.diff"
                script.append(f"cat > {patch_file} << 'EOL'\n{self.config['patch']}\nEOL")
                script.append(f"git apply {patch_file} || echo 'Failed to apply patch'")

        return script

    def construct(self):
        env_script = self._build_repo_base_env()
        eval_script = self._build_eval_script()
        
        instance_id = self.config.get("instance_id", "unknown")
        target_dir = os.path.join(self.config["workdir"], f"{self.config['repo'].split('/')[1]}_{instance_id}")
        
        self.task = Task(instance_id=instance_id,
                         env_script=env_script,
                         eval_script=eval_script,
                         patch=self.config["patch"],
                         target_dir=target_dir,
                         output_dir=self.config["output_dir"],
                         apply_patch=self.config["apply_patch"])

    def parse_test_results(self, output):
        passed_pattern = r"test ([\w:]+) \.\.\. ok"
        failed_pattern = r"test ([\w:]+) \.\.\. FAILED"
        ignored_pattern = r"test ([\w:]+) \.\.\. ignored"
        
        passed_tests = re.findall(passed_pattern, output)
        failed_tests = re.findall(failed_pattern, output)
        ignored_tests = re.findall(ignored_pattern, output)
        
        test_results = {
            "passed": passed_tests,
            "failed": failed_tests,
            "ignored": ignored_tests,
            "failure_details": {}
        }
        
        for test in failed_tests:
            regex_pattern = rf"---- {re.escape(test)} stdout ----\n(.*?)(?:\n\nfailures:|\n\ntest result:|$)"
            failure_details = re.search(regex_pattern, output, re.DOTALL)
            if failure_details:
                test_results["failure_details"][test] = failure_details.group(1).strip()
        
        return test_results

    def check_env(self):
        if not os.path.exists(self.task.target_dir):
            raise Exception(f'Repo {self.task.target_dir} does not exist. Please check.')
        if not os.path.exists(self.config["workdir"]):
            raise Exception(f'Workdir {self.config["workdir"]} does not exist. Please check.')

    def run_ci(self):
        """Run tests and save results to log file"""
        try:
            logger.info(f"Starting CI run for {self.config['repo']} (ID: {self.config.get('instance_id', 'unknown')})")

            task = self.task
            self._execute_scripts(cwd=task.target_dir)
            logger.info(f"Running cargo test in {task.target_dir}")
            
            result = subprocess.run(
                ["cargo", "test"],
                cwd=task.target_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Cargo test completed with return code: {result.returncode}")
            
            test_results = self.parse_test_results(result.stdout)
            output = {"unit_test": {
                "returncode": result.returncode,
                "test_results": test_results
            }}
            
            return output

        except Exception as e:
            logger.error(f"Task failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"unit_test": {
                "returncode": 1,
                "error": str(e),
                "test_results": {"passed": [], "failed": [], "ignored": [], "failure_details": {}}
            }}
            
    def _execute_scripts(self, cwd="~"):
        """Execute environment setup and evaluation scripts, hide output"""
        # Ensure each repository uses unique script file paths
        repo_name = self.config["repo"].split("/")[1]
        instance_id = self.config.get("instance_id", "unknown")
        script_dir = os.path.join(self.config["workdir"], f"{repo_name}_{instance_id}")
        
        logger.info(f"Creating script directory: {script_dir}")
        # Create script directory
        os.makedirs(script_dir, exist_ok=True)
        
        # Execute environment setup script
        env_script_path = os.path.join(script_dir, "env_setup.sh")
        with open(env_script_path, 'w') as f:
            f.write('\n'.join(self.task.env_script))
        
        logger.info("Executing environment setup script")
        subprocess.run(
            ['chmod', '+x', env_script_path], 
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        subprocess.run(
            ['bash', env_script_path], 
            check=True,
            # cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Execute evaluation script
        eval_script_path = os.path.join(script_dir, "eval.sh")
        with open(eval_script_path, 'w') as f:
            f.write('\n'.join(self.task.eval_script))

        self.check_env()

        logger.info("Executing evaluation script")
        subprocess.run(
            ['chmod', '+x', eval_script_path], 
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        subprocess.run(
            [eval_script_path], 
            check=True,
            # cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

class DockerCITool(CIToolBase):
    def __init__(self, config):
        super().__init__(config)
        self.construct()

    def construct(self):
        pass


class ActCITool(CIToolBase):
    def __init__(self, config):
        super().__init__(config)
        self.act_list_path = 'act_list.txt'
        self.apply_patch = self.config["apply_patch"]
        self.cloned_repo_path = self.config["repo"].split("/")[1] + "__" + self.config["merge_commit"]
        self.ci_dict = dict()
        self.result_lock = threading.Lock()
        self.result_list = []

        self.construct()

    # TODO(wdxu): make these two functions to be public methods.
    def _build_repo_base_env(self):
        script = ["#!/bin/bash"]
        
        repo_dir_name = self.config["repo"].replace('/', '__')
        instance_id = self.config.get("instance_id", "unknown")
        src_path = os.path.join(self.config["src_folder"], repo_dir_name)
        dst_path = os.path.join(self.config["workdir"], f"{self.config['repo'].split('/')[1]}_{instance_id}")
        
        script.append(f"mkdir -p {dst_path}")
        script.append(f"cp -r {src_path}/. {dst_path}/")

        return script

    def _build_eval_script(self):
        instance_id = self.config.get("instance_id", "unknown")
        target_dir = os.path.join(self.config["workdir"], f"{self.config['repo'].split('/')[1]}_{instance_id}")

        script = ["#!/bin/bash", 
                  f"cd {target_dir}",
                 ]
        
        script.append("git stash -u || true")
        
        if "merge_commit" in self.config and self.config["merge_commit"]:
            script.append("git checkout " + self.config["merge_commit"])
            
            # Apply test_patch if it exists
            if self.config.get("test_patch"):
                test_patch_file = f"{target_dir}/test_patch.diff"
                script.append(f"cat > {test_patch_file} << 'EOL'\n{self.config['test_patch']}\nEOL")
                script.append(f"git apply {test_patch_file} || echo 'Failed to apply test_patch'")
            
            # Apply patch only if apply_patch is specified
            if self.config.get("apply_patch", False) and self.config.get("patch"):
                patch_file = f"{target_dir}/patch.diff"
                script.append(f"cat > {patch_file} << 'EOL'\n{self.config['patch']}\nEOL")
                script.append(f"git apply {patch_file} || echo 'Failed to apply patch'")

        return script


    def _get_ci_job_name_id_dict(self, target_dir):
        def _extract_jobs(filename):
            jobs = {}
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("Stage"):
                        continue
                    columns = re.split(r'\s{2,}', line)
                    if len(columns) >= 3:
                        job_id = columns[1]
                        job_name = columns[2]
                        jobs[job_name] = job_id
            return jobs

        script = ["#!/bin/bash"]
        script.extend(["cd " + target_dir])
        script.extend([f"act --list > {self.act_list_path}"])
        os.system("\n".join(script))
        # only absolute path? 
        act_list_path = os.path.join(target_dir, self.act_list_path)
        self.ci_dict = _extract_jobs(os.path.expanduser(act_list_path))
        os.system("rm " + act_list_path)
                    
    def _process_act_output(self, stdout):
        results = []
        for line in stdout.split('\n'):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                result = {
                    'dryrun': data.get('dryrun', ''),
                    'job': data.get('job', ''),
                    'jobID': data.get('jobID', ''),
                    'level': data.get('level', ''),
                    'matrix': data.get('matrix', ''),
                    'msg': data.get('msg', ''),
                    'raw_output': data.get('raw_output', ''),
                    'stage': data.get('stage', ''),
                    'step': data.get('step', ''),
                    'stepID': data.get('stepID', ''),
                    'stepResult': data.get('stepResult', ''),
                    'time': data.get('time', ''),
                }
                results.append(result)
            except json.JSONDecodeError:
                continue
        return results

    def _run_act_with_lock(self, ci, target_dir, order, pool):
        value = self.ci_dict.get(ci[0])
        if value is not None:
            port = pool.acquire_port()
            path = self.config["output_dir"] + "/" + \
                   self.task.instance_id + "_"  + \
                   value + "_" + \
                   order + "_output.json"
            # Do not ignore the existing results.
            # if os.path.exists(path):
            #     print(f"path exists: {path}. Ignore...")
            #     return
            # print(target_dir)
            # print(os.path.join(target_dir, ci[1]))
            logger.info("Run Act with command: " + "act " + "-j " + value + " " \
                                            "-P " + "ubuntu-latest=catthehacker/ubuntu:full-latest " + \
                                            "--artifact-server-port " + str(port) + " " +\
                                            "--artifact-server-addr " + "0.0.0.0" + " " +\
                                            "--artifact-server-path " + f"./act/{port}" + " " +\
                                            "-W " + os.path.join(target_dir, ci[1]) + " " +\
                                            "--json")

            process = subprocess.Popen(["act", "-j", value,
                                        "-P", "ubuntu-latest=catthehacker/ubuntu:full-latest",
                                        "--artifact-server-port", str(port),
                                        "--artifact-server-addr", "0.0.0.0", 
                                        "--artifact-server-path", f"./act/{port}",
                                        "-W", os.path.join(target_dir, ci[1]),
                                        "--json"], 
                                    cwd=target_dir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            stdout, stderr = process.communicate()
            result = {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": process.returncode,
                "processed_output": self._process_act_output(stdout)
            }
            result_path = os.path.join(target_dir, path) 
            if not os.path.exists(os.path.dirname(result_path)):
                os.makedirs(os.path.dirname(result_path))
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            self.result_lock.acquire()
            self.result_list.append(result)
            self.result_lock.release()

    @staticmethod
    def _process_result(result_list: list[str]) -> dict:
        processed_result = {}

        for result in result_list:
            if type(result) == str:
                result_json = json.loads(result)
            else:
                result_json = result
            for processed_output in result_json["processed_output"]:
                job_id = processed_output["jobID"]
                if job_id not in processed_result:
                    processed_result[job_id] = {
                        "returncode": result_json["returncode"],
                        "test_results": {
                            "passed": [],
                            "failed": [],
                            "ignored": [],
                        }
                    }
                if processed_output["stepResult"] == "success" and processed_output["stage"] != "":
                    processed_result[job_id]["test_results"]["passed"].append({
                                                       "stage": processed_output["stage"],
                                                       "step": processed_output["step"],
                                                       "stepID": processed_output["stepID"]})
                elif processed_output["stepResult"] == "failure":
                    processed_result[job_id]["test_results"]["failed"].append({
                                                       "stage": processed_output["stage"],
                                                       "step": processed_output["step"],
                                                       "stepID": processed_output["stepID"]})
                elif processed_output["stepResult"] == "skipped":
                    processed_result[job_id]["test_results"]["ignored"].append({
                                                        "stage": processed_output["stage"],
                                                        "step": processed_output["step"],
                                                        "stepID": processed_output["stepID"]})
        return processed_result

    def check_env(self):
        if not os.path.exists(self.task.target_dir):
            raise Exception(f'Repo {self.task.target_dir} does not exist. Please check.')
        if not os.path.exists(self.config["workdir"]):
            raise Exception(f'Workdir {self.config["workdir"]} does not exist. Please check.')

    def run_ci(self, pool):
        task = self.task
        run_script("\n".join(task.env_script))
        self.check_env()
        run_script("\n".join(task.eval_script))

        logger.info(f"Starting CI run for {self.config['repo']} (ID: {self.config.get('instance_id', 'unknown')})")

        self._get_ci_job_name_id_dict(task.target_dir)
        logger.info(f'Collected CI job name and id dict: {self.ci_dict}')
        threads = []
        for ci in self.config["ci_name_list"]:
            thread = threading.Thread(
                target=lambda ci=ci: self._run_act_with_lock(ci, task.target_dir, "merged", pool)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # os.system("rm -rf " + task.target_dir)
        result = ActCITool._process_result(self.result_list)
        logger.info(f"CI run completed for {self.config['repo']} (ID: {self.config.get('instance_id', 'unknown')})")
        return result

    def construct(self):
        env_script = self._build_repo_base_env()
        eval_script = self._build_eval_script()

        instance_id = self.config.get("instance_id", "unknown")
        target_dir = os.path.join(self.config["workdir"], f"{self.config['repo'].split('/')[1]}_{instance_id}")
        
        self.task = Task(instance_id=instance_id,
                         env_script=env_script,
                         eval_script=eval_script,
                         patch=self.config["patch"],
                         target_dir=target_dir,
                         output_dir=self.config["output_dir"],
                         apply_patch=self.config["apply_patch"])

EVAL_HANDLER = {
    "cargo": CargoCITool,
    "docker": DockerCITool,
    "act": ActCITool
}

RUST_BASE_ENV={
    "vectordotdev/vector": ["protobuf-compiler", "libsasl2-dev"]
}

RUST_INSTALL = ["if ! command -v rustc >/dev/null 2>&1; then",
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                "source \"$HOME/.cargo/env\"", "fi"]


if __name__ == '__main__':
    # inputs = ''
    # with open(os.path.join(os.environ["SWING_TESTBED_PATH"], "cplee__github-actions-demo-1_test_merged_output.json"), 'r') as f:
    #     while x := f.readline():
    #         inputs += x
    #     result = ActCITool._process_result([inputs])
    #     for each in result:
    #         print(each, result[each])
    # exit()

    from swebench.harness.utils import PortPool
    port_pool = PortPool([i for i in range(50505, 52505)])

    # Comment(wdxu): fake data for test only.
    # act = ActCITool({"act_path": "/mnt/Data/wdxu/github/act/bin/act", \
    #                  "instance_id": "cplee__github-actions-demo-1", \
    #                  "repo": "cplee/github-actions-demo", \
    #                  "base_commit": "2dcabf3769c2613687310c7b71b89af681e8ee50", \
    #                  "merge_commit": "2dcabf3769c2613687310c7b71b89af681e8ee50", \
    #                  "patch": "", \
    #                  "apply_patch": True, \
    #                  "src_folder": os.environ["SWING_TESTBED_PATH"], \
    #                  "workdir": os.environ["SWING_TESTBED_PATH"], \
    #                  "ci_name_list": [["test", ".github/workflows/main.yml"]], \
    #                  "output_dir": os.environ["SWING_TESTBED_PATH"]})
    act = ActCITool({"act_path": "/mnt/Data/wdxu/github/act/bin/act",
                     "instance_id": "rustzx__rustzx-84",
                     "repo": "rustzx/rustzx",
                     "base_commit": "53cfe0985162dc3e7f6f64fee77a67e3c08a1b9a",
                     "merge_commit": "53cfe0985162dc3e7f6f64fee77a67e3c08a1b9a",
                     "patch": "",
                     "src_folder": "/mnt/Data/wdxu/github/Swing-Bench/testbed",
                     "output_dir": "logs",
                     "workdir": "/mnt/Data/wdxu/github/Swing-Bench/testbed/rustzx__rustzx-84_0842a35c-2520-48a4-93c5-d320350242a6",
                     "apply_patch": True,
                     "ci_name_list": [['build', '.github/workflows/ci.yml'], ['unit_tests', '.github/workflows/test-rustzx-z80.yml']]})

    result = act.run_ci(port_pool)
    print(result)
    # with open('./result.log', 'w') as f:
    #     f.write(str(result))