from abc import ABC, abstractmethod
from pathlib import Path
from swebench.harness.agent.swingbench import SwingbenchInstance
import json
import logging
from argparse import ArgumentParser
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

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
        repo.replace('/', '_') / 
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
            "hits": []
        }
        
        for hit in hits:
            raw_doc = json.loads(searcher.doc(hit.docid).raw())
            results["hits"].append({
                "docid": hit.docid,
                "score": hit.score,
                "contents": raw_doc.get("contents", ""),
                "relative_path": raw_doc.get("instance_id", "")
            })
            
        return results
        
    except Exception as e:
        logger.error(f"Search failed for instance {instance.instance_id}")
        logger.error(str(e))
        return None

class Retriever(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Retriever is not implemented yet.")

    @abstractmethod
    def retrieve(self, instance: SwingbenchInstance):
        raise NotImplementedError("Retrieve is not implemented yet.")

class BM25DiskRetriever(Retriever):
    def __init__(self, index_dir: str, document_encoding_style: str = "file_name_and_contents"):
        self.index_dir = Path(index_dir)
        self.document_encoding_style = document_encoding_style

    def retrieve(self, instance: SwingbenchInstance):
        results = search_instance(
            instance,
            self.index_dir,
            self.document_encoding_style,
            k=1
        )
        # TODO(wdxu): need some reduce strategies
        
        return results
    
def main(debug: bool = False):
    """_summary_

    Returns:
        dict: {instance_id: retrieved results}
        
        JSONL format:
        {
            "instance_id": instance_id,
            "hits": [
                {
                    "docid": docid (path/file.ext),
                    "score": score,
                    "contents": contents,
                    "relative_path": relative_path
                },
                ...
            ]
        }
    """
    parser = ArgumentParser()
    parser.add_argument("--instances_file", type=str, required=True,
                      help="Path to the dataset.json file")
    parser.add_argument("--index_dir", type=str, required=True,
                      help="Root directory containing indexes")
    parser.add_argument("--document_encoding_style",
                      choices=["file_name_and_contents", "file_name_and_documentation"],
                      default="file_name_and_contents")
    parser.add_argument("--output_file_for_debug", type=str, default=None,
                      help="Output file to store search results")
    args = parser.parse_args()
    
    instances = []
    with open(args.instances_file) as f:
        for line in f:
            instances.append(SwingbenchInstance(**json.loads(line)))
    
    index_root = Path(args.index_dir)
    if debug:
        with open(args.output_file_for_debug, "w") as out_f:
            for instance in tqdm(instances, desc="Searching"):
                results = search_instance(
                    instance,
                    index_root,
                    args.document_encoding_style
                )
                if results:
                        print(json.dumps(results), flush=True)
    else:
        results_dir = {}
        for instance in tqdm(instances, desc="Searching"):
            results = search_instance(
                instance,
                index_root,
                args.document_encoding_style
            )
            if results:
                results_dir[instance.instance_id] = results
        return results_dir
    
if __name__ == "__main__":
    """
    python swing_search_index.py \
        --instances_file /mnt/Data/wdxu/github/Swing-Bench/tmpdata/dataset.json \
        --index_dir /mnt/Data/wdxu/github/Swing-Bench/tmpdata/indexes \
        --document_encoding_style file_name_and_contents \
        --output_file_for_debug results.jsonl
    """
    main(debug=True)
    # result = main(debug=False)
    # for each in result:
    #     print(each, result[each])