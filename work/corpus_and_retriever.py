import logging
import pickle
import faiss
import PyPDF2
import docx
import re
from sentence_transformers import SentenceTransformer
import torch
from work.Abstract_Retriver_Corpus import CustomRetrieverAndCorpus


class CorpusAndRetriever(CustomRetrieverAndCorpus):
    def __init__(self, model_name: str = "intfloat/e5-large-v2", index_path: str = "", doc_index_path: str = "",
                 index_save_path: str = "retrieve_index.faiss", doc_save_index_path: str = "documents_index.faiss"):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.doc_save_index_path = doc_save_index_path
        self.documents_index = {}
        self.current_index = 0
        self.encoder = None
        self.load_model()
        self.create_index(index_path)

        if index_path:
            self.index_save_path = index_path
        if doc_index_path:
            self.doc_save_index_path = doc_index_path

        if doc_index_path:
            self.load_documents_index(doc_index_path)

    def load_model(self):
        """
        Load the encoder using SentenceTransformer.
        """
        logging.info("Loading Ranker for RAG...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(self.model_name).to(device)
        logging.info("Ranker loaded successfully")

    def encode_chunks(self, chunks: list, batch_size: int = 16):
        """
        Encode the chunks using the class encoder.
        :param chunks: List of chunks to encode.
        :param batch_size: Batch size for encoding.
        :return: List of encoded chunks.
        """
        logging.info("Encoding chunks...")
        embeddings = self.encoder.encode(chunks, normalize_embeddings=True, batch_size=batch_size)
        logging.info("chunks Encoded successfully...")

        return embeddings

    def create_index(self, index_path: str = ""):
        """Creates an empty FAISS index for vectors of a specified dimension."""
        if index_path:
            self.index = faiss.read_index(index_path)
            logging.info(f"Index loaded from {index_path}")
            #print(f"Number of documents in the index: {self.index.ntotal}")
            return
        index = faiss.IndexFlatL2(self.encoder.get_sentence_embedding_dimension())
        # Wrap the base index with IndexIDMap
        self.index = faiss.IndexIDMap(index)

    def load_documents_index(self, doc_index_path: str):
        """ Load the documents index (dictionary) from a file using pickle. """
        try:
            with open(doc_index_path, "rb") as f:
                data = pickle.load(f)
                self.documents_index = data["documents_index"]
                self.current_index = data["current_index"]
            logging.info(f"Documents index loaded from {doc_index_path}")
        except FileNotFoundError:
            logging.warning(f"Documents index file not found at {doc_index_path}. Starting with an empty index.")

    def save_index(self):
        """
        Save the index to a file.
        :param index_path: Path to save the index.
        """
        faiss.write_index(self.index, self.index_save_path)
        logging.info(f"Index saved to {self.index_save_path}")

    def save_documents_index(self):
        """ Save the documents index (dictionary) to a file using pickle. """
        with open(self.doc_save_index_path, "wb") as f:
            pickle.dump({"documents_index": self.documents_index, "current_index": self.current_index}, f)
        logging.info(f"Documents index saved to {self.doc_save_index_path}")

    def process_files(self, file_path_list: list, overlap: int = 25):
        """
        Process the file at the given path and split it into chunks of 250 words with overlap.
        :param file_path_list: List of file paths to process.
        :param overlap: Number of words to overlap between chunks.
        :return: List of chunks containing 250 words each.
        """
        chunks = []
        for file_path in file_path_list:
            # Check file type and extract text accordingly
            if file_path.endswith('.pdf'):
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text() is not None])

            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                text = " ".join([para.text for para in doc.paragraphs if para.text])

            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

            else:
                raise ValueError("Unsupported file format")

            # Clean and split text into words
            words = self.clean_text(text)
            words = re.split(r'\s+', text.strip())

            # Split documents into chunks and add them to index
            new_chunks = self.split_into_chunks(words, overlap)

            chunks.extend(new_chunks)

        return chunks

    def add_chunks_to_index(self, chunks: list):
        """
        Add chunks to the FAISS index along with associated IDs after normalizing them.
        :param chunks: List of chunks to add to the index.
        """
        indices = []
        for chunk in chunks:
            self.documents_index[self.current_index] = chunk
            indices.append(self.current_index)
            self.current_index += 1
        return indices

    def add_documents(self, file_path_list: list):
        """
        Adds vectors to the FAISS index along with associated IDs after normalizing them.

        Parameters:
        :param file_path_list: List of file paths to process.
        ids (np.array): 1D NumPy array containing the IDs for the documents
        """
        # Process files into chunks:
        chunks = self.process_files(file_path_list)

        # Add chunks to the index
        indices = self.add_chunks_to_index(chunks)

        # Encode the chunks
        docs_embeddings = self.encode_chunks(chunks)

        # Add vectors and their IDs to the index
        self.index.add_with_ids(docs_embeddings, indices)

        # Save the new index and documents index
        self.save_index()
        self.save_documents_index()

    def clean_text(self, text: str):
        """
        Cleans extracted legal text by removing file paths, metadata, extra whitespace, and formatting noise.
        """
        # Remove file paths (e.g., "Y:\SGML\262005.XXX...")
        text = re.sub(r"Y:\\SGML\\.*?\s", "", text)

        # Remove metadata like "VerDate Sep<11>2014 12:59 Jun 10, 2024 Jkt 262005 PO 00000 Frm 00263 Fmt 8008 Sfmt 8008"
        text = re.sub(r"VerDate\s\w+\s\d{1,2},\s\d{4}\s\d{2}:\d{2}.*?Frm\s\d+\sFmt\s\d+\sSfmt\s\d+", "", text)

        # Remove section numbers and reserved sections (e.g., "401–414 [Reserved]")
        text = re.sub(r"\d{3}–\d{3} \[Reserved\]", "", text)

        # Remove extra spaces, newlines, and tabs
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def split_into_chunks(self, words, overlap: int = 50):
        """
        Split the list of words into chunks of 250 words with words overlap.
        :param words: List of words to split into chunks.
        :param overlap: Number of words to overlap between chunks.
        :return: List of chunks containing 250 words each.
        """
        # Generate chunks of 250 words with overlap
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:min(i + 250, len(words))])
            chunks.append(chunk)
            i += (250 - overlap)  # Move x words ahead (x = 250 - overlap)

        return chunks

    def retrieve(self, query: str, num_results: int = 5):
        """
        Receives a query and returns the k closest chunks to it by their embeddings, sorted so the closest is last.

        Parameters:
        query (str): Query string to search for.
        num_results (int): Number of closest chunks to retrieve.

        Returns:
        List of the chunks sorted from least similar to most similar.
        """
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)
        D, I = self.index.search(query_embedding, num_results)  # Search for k nearest neighbors

        # Retrieve the chunks sorted by increasing distance, return only the chunks
        sorted_chunks = []
        for idx in I[0]:
            sorted_chunks.append(self.documents_index[idx])
        sorted_chunks.reverse()  # Reverse to make the closest chunk last
        return sorted_chunks

