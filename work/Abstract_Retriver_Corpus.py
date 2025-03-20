from abc import ABC, abstractmethod


class CustomRetrieverAndCorpus(ABC):
    """
        Abstract base class for document retrival and corpus manager.
    """
    @abstractmethod
    def load_model(self):
        """
        Load the encoder using SentenceTransformer.
        """
        pass

    @abstractmethod
    def encode_chunks(self, chunks: list):
        """
        Encode the chunks using the class encoder.
        :param chunks: List of chunks to encode.
        :return: List of encoded chunks.
        """
        pass

    @abstractmethod
    def create_index(self):
        """Creates an empty FAISS index for vectors of a specified dimension."""
        pass

    @abstractmethod
    def process_files(self, file_path_list: list, overlap: int = 25):
        """
        Process the file at the given path and split it into chunks of 250 words with overlap.
        :param file_path_list: List of file paths to process.
        :param overlap: Number of words to overlap between chunks.
        :return: List of chunks containing 250 words each.
        """
        pass

    @abstractmethod
    def add_chunks_to_index(self, chunks: list):
        """
        Add chunks to the FAISS index along with associated IDs after normalizing them.
        :param chunks: List of chunks to add to the index.
        """
        pass

    @abstractmethod
    def add_documents(self, file_path_list: list):
        """
        Adds vectors to the FAISS index along with associated IDs after normalizing them.
        """
        pass

    @abstractmethod
    def split_into_chunks(self, words, overlap: int = 50):
        """
        Split the list of words into chunks of 250 words with words overlap.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, num_results: int = 5):
        """
        Use the DataProcessor to retrieve relevant document chunks based on the query.
        """
        pass