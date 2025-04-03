# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from argparse import ArgumentParser
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

from swebench.harness.constants.swing_constants import SwingbenchInstance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def search_instance(
    instance: SwingbenchInstance,
    index_root: Path,
    document_encoding_style: str,
    k: int = 20
) -> dict:
    repo = instance.repo
    commit = instance.base_commit
    query = instance.problem_statement
    
    index_path = (
        index_root / 
        repo.replace('/', '__') / 
        document_encoding_style /
        commit /
        "index"
    )
    
    if not index_path.exists():
        logger.error(f"Index not found for {repo} at commit {commit}")
        return None
        
    try:
        searcher = LuceneSearcher(index_path.as_posix())
        cutoff = len(query)
        
        while True:
            try:
                hits = searcher.search(
                    query[:cutoff],
                    k=k,
                    remove_dups=True,
                )
                break
            except Exception as e:
                if "maxClauseCount" in str(e):
                    cutoff = int(round(cutoff * 0.8))
                    continue
                else:
                    raise e
        
        results = {
            "instance_id": instance.instance_id,
            "hits": [{"docid": hit.docid, "score": hit.score} for hit in hits]
        }
        return results
        
    except Exception as e:
        logger.error(f"Search failed for instance {instance.instance_id}")
        logger.error(str(e))
        return None

def main():
    parser = ArgumentParser()
    parser.add_argument("--instances_file", type=str, required=True,
                      help="Path to the dataset.json file")
    parser.add_argument("--index_dir", type=str, required=True,
                      help="Root directory containing indexes")
    parser.add_argument("--document_encoding_style",
                      choices=["file_name_and_contents", "file_name_and_documentation"],
                      default="file_name_and_contents")
    parser.add_argument("--output_file", type=str, required=True,
                      help="Output file to store search results")
    args = parser.parse_args()
    
    instances = []
    with open(args.instances_file) as f:
        for line in f:
            instances.append(SwingbenchInstance(**json.loads(line)))
    
    index_root = Path(args.index_dir)
    with open(args.output_file, "w") as out_f:
        for instance in tqdm(instances, desc="Searching"):
            results = search_instance(
                instance,
                index_root,
                args.document_encoding_style
            )
            if results:
                print(json.dumps(results), file=out_f, flush=True)

if __name__ == "__main__":
    # python search_index.py \
    # --instances_file /mnt/Data/wdxu/github/Swing-Bench/tmpdata/dataset.json \
    # --index_dir ./indexes \
    # --document_encoding_style file_name_and_contents \
    # --output_file results.jsonl
    main()