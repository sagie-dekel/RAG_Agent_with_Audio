from abc import ABC, abstractmethod

from work.Abstract_Retriver_Corpus import CustomRetrieverAndCorpus


class CustomGraphRAG(ABC):
    """
        Abstract base class for RAG graph
    """
    def __init__(self, corpus_retriever: CustomRetrieverAndCorpus):
        self.corpus_retriever = corpus_retriever

    @abstractmethod
    def activate_RAG_graph(self, query: str, k: int = 5):
        """
        Perform retrieval and then generate a response based on the RAG graph policy.
        """
        pass

