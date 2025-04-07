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