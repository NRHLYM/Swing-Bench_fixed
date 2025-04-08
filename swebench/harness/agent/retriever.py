from abc import ABC, abstractmethod
from pathlib import Path
from swebench.harness.constants.swing_constants import SwingbenchInstance
from swebench.inference.make_datasets.swing_search_index import search_instance


class Retriever:
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


if __name__ == "__main__":
    import swing_utils

    retriever = BM25DiskRetriever(index_dir="/mnt/Data/wdxu/github/Swing-Bench/tmpdata/indexes")
    dataset_jsonl_path = '/mnt/Data/wdxu/github/Swing-Bench/tmpdata/dataset.json'
    dataset = swing_utils.load_swingbench_dataset(dataset_jsonl_path)
    
    for instance in dataset:
        results = retriever.retrieve(instance)
        print(results)
