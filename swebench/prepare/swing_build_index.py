# -*- coding: utf-8 -*-

import json
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Set
from git import Repo
from tqdm import tqdm
import shutil

from swebench.harness.constants.swing_constants import SwingbenchInstance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def extract_repo_commits(dataset_path: str) -> Dict[str, Set[str]]:
    repo_commits: Dict[str, Set[str]] = {}
    
    with open(dataset_path) as f:
        for line in f:
            instance = SwingbenchInstance(**json.loads(line))
            if instance.repo not in repo_commits:
                repo_commits[instance.repo] = set()
            repo_commits[instance.repo].add(instance.base_commit)
    
    return repo_commits


def clone_repo(repo: str, root_dir: str, token: str) -> Path:
    repo_dir = Path(root_dir) / f"repo_{repo.replace('/', '_')}"
    
    if not repo_dir.exists():
        repo_url = f"https://{token}@github.com/{repo}.git"
        logger.info(f"Cloning {repo}")
        Repo.clone_from(repo_url, repo_dir)
    
    return repo_dir


def build_documents(repo_dir: Path, commit: str, document_encoding_func) -> Dict[str, str]:
    documents = {}
    
    repo = Repo(repo_dir)
    repo.git.reset("--hard", commit)
    repo.git.clean("-fdxq")
    
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.startswith("."):
                continue
            file_path = Path(root) / file
            relative_path = str(file_path.relative_to(repo_dir))
            try:
                text = document_encoding_func(file_path, relative_path)
                documents[relative_path] = text
            except Exception as e:
                logger.error(f"Error processing {relative_path}: {e}")
                continue
    
    return documents


def file_name_and_contents(filename: Path, relative_path: str) -> str:
    text = relative_path + "\n"
    with open(filename) as f:
        text += f.read()
    return text


def file_name_and_documentation(filename: Path, relative_path: str) -> str:
    import ast
    text = relative_path + "\n"
    try:
        with open(filename) as f:
            node = ast.parse(f.read())
        data = ast.get_docstring(node)
        if data:
            text += f"{data}"
        for child_node in ast.walk(node):
            if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                data = ast.get_docstring(child_node)
                if data:
                    text += f"\n\n{child_node.name}\n{data}"
    except Exception as e:
        logger.error(f"Failed to parse file {filename}: {e}")
        with open(filename) as f:
            text += f.read()
    return text


DOCUMENT_ENCODING_FUNCTIONS = {
    "file_name_and_contents": file_name_and_contents,
    "file_name_and_documentation": file_name_and_documentation,
}


def build_repo_index(
    repo: str,
    commits: Set[str],
    root_dir: str,
    repo_root_dir: str,
    document_encoding_style: str,
    python: str,
):
    document_encoding_func = DOCUMENT_ENCODING_FUNCTIONS[document_encoding_style]
    
    index_root = Path(root_dir) / repo.replace('/', '__') / document_encoding_style
    index_root.mkdir(parents=True, exist_ok=True)
    
    repo_dir = os.path.join(repo_root_dir, repo.replace('/', '__'))
    
    for commit in tqdm(commits, desc=f"Building indexes for {repo}"):
        commit_index_path = index_root / commit
        if commit_index_path.exists():
            logger.info(f"Index for commit {commit} already exists, skipping...")
            continue
            
        try:
            documents = build_documents(repo_dir, commit, document_encoding_func)
            
            documents_path = commit_index_path / "documents.jsonl"
            documents_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(documents_path, "w") as docfile:
                for relative_path, contents in documents.items():
                    print(
                        json.dumps({"id": relative_path, "contents": contents}),
                        file=docfile,
                        flush=True,
                    )
            
            index_path = commit_index_path / "index"
            cmd = [
                python,
                "-m",
                "pyserini.index",
                "--collection", "JsonCollection",
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", "2",
                "--input", documents_path.parent.as_posix(),
                "--index", index_path.as_posix(),
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully built index for commit {commit}")
            
        except Exception as e:
            logger.error(f"Failed to build index for commit {commit}")
            logger.error(str(e))
            continue

    if os.path.exists(repo_dir):
        print(f'Remove finished {repo_dir}.')
        shutil.rmtree(repo_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to the dataset.json file")
    parser.add_argument("--output_dir", type=str, default="./indexes",
                      help="Directory to store the indexes")
    parser.add_argument("--document_encoding_style",
                      choices=DOCUMENT_ENCODING_FUNCTIONS.keys(),
                      default="file_name_and_contents")
    parser.add_argument("--repo_root_dir", type=str, default="./testbed",
                      help="Directory to store the repos")
    args = parser.parse_args()
    
    repo_commits = extract_repo_commits(args.dataset_path)
    
    python = subprocess.run("which python", shell=True, capture_output=True)
    python = python.stdout.decode("utf-8").strip()
    
    for repo, commits in tqdm(repo_commits.items(), desc="Processing repositories"):
        build_repo_index(
            repo=repo,
            commits=commits,
            root_dir=args.output_dir,
            repo_root_dir=args.repo_root_dir,
            document_encoding_style=args.document_encoding_style,
            python=python,
        )


if __name__ == "__main__":
    """
    python swebench/prepare/swing_build_index.py \
        --dataset_path tmpdata/Swing-Dataset/Swing-Rust/all_tasks.jsonl \
        --repo_root_dir ./testbed \
        --output_dir tmpdata/indexes/
    """
    main()
