from datasets import load_dataset
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

dataset = load_dataset("SwingBench/SWE-Rust", split="train")

print(dataset)

def clone_repo(repo):
    git_url = f'https://github.com.psmoe.com/{repo}.git'
    repo_path = f'/raid/rust-repos/{repo.replace("/", "__")}'
    attempts = 0
    MAX_ATTEMPTS = 1
    if os.path.exists(repo_path) and len(os.listdir(repo_path)) > 1:
        return
    else:
        os.system(f'rm -rf {repo_path}')
    while attempts < MAX_ATTEMPTS:
        try:
            print(f'Cloning {git_url} into {repo_path} (Attempt {attempts + 1})...')
            subprocess.run(['git', 'clone', git_url, repo_path], check=True)
            print(f'Successfully cloned {git_url}')
            break
        except Exception as e:
            attempts += 1
            print(f'Error cloning {git_url}: {e}. Retrying... ({attempts}/{MAX_ATTEMPTS})')
    if attempts == MAX_ATTEMPTS:
        print(f'Failed to clone {git_url} after {MAX_ATTEMPTS} attempts.')


with ThreadPoolExecutor(max_workers=15) as executor:
    executor.map(clone_repo, dataset['repo'])
