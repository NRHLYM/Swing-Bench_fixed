import subprocess
import re
import os
import tempfile
import threading
from queue import Queue
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
    id: str
    env_script: list[str]
    eval_script: list[str]
    patch: str
    target_dir: str
    output_dir: str
    previous_eval_script: list[str] = None

class CIToolBase:
    def __init__(self, config):
        self.config = config

    def construct(self):
        pass
    
    def run_ci(self, log_file):
        pass

class CargoCITool(CIToolBase):
    def __init__(self, config):
        super().__init__(config)
        self.construct()

    def _build_repo_base_env(self):
        script = ["#!/bin/bash"]
        
        repo_dir_name = self.config["repo"].replace('/', '__')
        instance_id = self.config.get("id", "unknown")
        src_path = os.path.join(self.config["src_folder"], repo_dir_name)
        dst_path = os.path.join(self.config["workdir"], f"{self.config['repo'].split('/')[1]}_{instance_id}")
        
        script.append(f"mkdir -p {dst_path}")
        script.append(f"cp -r {src_path}/. {dst_path}/")

        return script

    def _build_eval_script(self):
        instance_id = self.config.get("id", "unknown")
        target_dir = os.path.join(self.config["workdir"], f"{self.config['repo'].split('/')[1]}_{instance_id}")
        
        script = ["#!/bin/bash", 
                  f"cd {target_dir}",
                 ]
        
        script.append("git stash -u || true")
        
        if "merge_commit" in self.config and self.config["merge_commit"]:
            script.append("git checkout " + self.config["merge_commit"])
            if self.config.get("apply_patch", False) and self.config.get("patch"):
                patch_file = f"{target_dir}/patch.diff"
                script.append(f"cat > {patch_file} << 'EOL'\n{self.config['patch']}\nEOL")

        return script

    def construct(self):
        env_script = self._build_repo_base_env()
        eval_script = self._build_eval_script()
        
        instance_id = self.config.get("id", "unknown")
        target_dir = os.path.join(self.config["workdir"], f"{self.config['repo'].split('/')[1]}_{instance_id}")
        
        self.task = Task("", env_script, eval_script, self.config["patch"], target_dir, self.config["output_dir"])

    def parse_test_results(self, log_file):
        with open(log_file, 'r') as f:
            output = f.read()
        
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

    def run_ci(self, log_file):
        """Run tests and save results to log file"""
        try:
            logger.info(f"Starting CI run for {self.config['repo']} (ID: {self.config.get('id', 'unknown')})")
            # Execute environment setup and evaluation scripts
            self._execute_scripts()
            
            task = self.task
            logger.info(f"Running cargo test in {task.target_dir}")
            with open(log_file, "w") as f:
                result = subprocess.run(
                    ["cargo", "test"], 
                    cwd=task.target_dir, 
                    stdout=f, 
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                    text=True
                )
            
            logger.info(f"Cargo test completed with return code: {result.returncode}")
            test_results = self.parse_test_results(log_file)
            
            output = {
                "returncode": result.returncode,
                "test_results": test_results
            }
            
            # Write processed results to JSON
            results_json_path = log_file + ".json"
            with open(results_json_path, 'w') as f:
                json.dump(output, f, indent=4)
            
            logger.info(f"Results saved to {results_json_path}")
            return output
        except Exception as e:
            logger.error(f"Task failed with exception: {str(e)}")
            # Record error to log file
            with open(log_file, "w") as f:
                f.write(f"Task failed with exception: {str(e)}")
            
            # Return error information
            return {
                "returncode": 1,
                "error": str(e),
                "test_results": {"passed": [], "failed": [], "ignored": [], "failure_details": {}}
            }

    def _execute_scripts(self):
        """Execute environment setup and evaluation scripts, hide output"""
        # Ensure each repository uses unique script file paths
        repo_name = self.config["repo"].split("/")[1]
        instance_id = self.config.get("id", "unknown")
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
            [env_script_path], 
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Execute evaluation script
        eval_script_path = os.path.join(script_dir, "eval.sh")
        with open(eval_script_path, 'w') as f:
            f.write('\n'.join(self.task.eval_script))
        
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
        self.cloned_repo_path = self.config["repo"].split("/")[1] + "_" + self.config["merge_commit"]
        self.ci_dict = dict()
        self.result_lock = threading.Lock()
        self.semaphore = threading.Semaphore(1)
        self.act_mq = Queue()
        self.construct()

    def _build_repo_base_env(self):
        script = ["#!/bin/bash"]
        script.extend(["cd " + self.config["workdir"],
                       "git clone https://github.com/" + self.config["repo"] + ".git " + self.cloned_repo_path])

        return script

    def _build_previous_eval_script(self):
        script = ["#!/bin/bash", 
                    "cd " + os.path.join(self.config["workdir"], self.cloned_repo_path),
                    "prev_commit=$(git rev-parse " + self.config["base_commit"] + "^)",
                    "git checkout $prev_commit"
                ]

        return script

    def _build_eval_script(self):
        script = ["#!/bin/bash", 
                    "cd " + os.path.join(self.config["workdir"], self.cloned_repo_path),
                    "git checkout " + self.config["merge_commit"]
                ]

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
        script.extend(["act --list > {}".format(self.act_list_path)])
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

    def _run_act_with_semaphore(self, ci, target_dir, order, pool):
        value = self.ci_dict.get(ci[0])
        if value is not None:
            port = pool.acquire_port()
            path = self.config["output_dir"] + "/" + self.task.id + "_"  + value + "_" + order + "_output.json"
            if os.path.exists(path):
                return
            process = subprocess.Popen(["act", "-j", value,
                                        "--artifact-server-port", str(port),
                                        "--artifact-server-addr", "0.0.0.0", 
                                        "--artifact-server-path", f"./act/{port}",
                                        "-W", ci[1],
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
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            self.act_mq.put(result)
    
    def run_ci(self, pool):
        task = self.task
        run_script("\n".join(task.env_script))
        run_script("\n".join(task.eval_script))

        if self.apply_patch:
            run_script()
                        
        self._get_ci_job_name_id_dict(task.target_dir)
        eval_result = []
        threads = []
        for ci in self.config["ci_name_list"]:
            thread = threading.Thread(
                target=lambda ci=ci: self._run_act_with_semaphore(ci, task.target_dir, "merged", pool)
            )
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        while not self.act_mq.empty():
            eval_result.append(self.act_mq.get())
        
        run_script("\n".join(task.previous_eval_script))
        if task.previous_eval_script:
            result = run_script(f"git apply {task.previous_eval_script}")
            if not result:
                print("Apply test patch successfully")
            else:
                print(f'Error when applying test patch: {result}')

        previous_eval_result = []
        threads = []
        for ci in self.config["ci_name_list"]:
            thread = threading.Thread(
                target=lambda ci=ci: self._run_act_with_semaphore(ci, task.target_dir, "based", pool)
            )
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
            
        while not self.act_mq.empty():
            previous_eval_result.append(self.act_mq.get())

        os.system("rm -rf " + task.target_dir)

        return [eval_result, previous_eval_result]

    def construct(self):
        env_script = self._build_repo_base_env()
        eval_script = self._build_eval_script()
        previous_eval_script = self._build_previous_eval_script()
        target_dir = os.path.join(self.config["workdir"],
                                  self.cloned_repo_path)
        self.task = Task(self.config["id"], env_script, eval_script, self.config["patch"], target_dir, self.config["output_dir"], previous_eval_script)


HANDLER = {
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
    # Comment(wdxu): fake data for test only.
    act = ActCITool({"act_path": "/mnt/Data/wdxu/github/act/bin/act", \
                     "repo": "cplee/github-actions-demo", \
                     "base_commit": "2dcabf3769c2613687310c7b71b89af681e8ee50", \
                     "merge_commit": "2dcabf3769c2613687310c7b71b89af681e8ee50", \
                     "patch": "patch_content", \
                     "workdir": "/home/wdxu/testbed", \
                     "output_dir": "output_dir"})
    result = act.run_ci(['test'])
    with open('./result.log', 'w') as f:
        f.write(str(result))