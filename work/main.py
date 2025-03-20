import logging

import torchaudio

from work.Abstract_Retriver_Corpus import CustomRetrieverAndCorpus
from work.RAG_manager import RAGManager
from work.corpus_and_retriever import CorpusAndRetriever
from work.graph_RAG import GraphRAG


def main():
    # Reconfigure logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # Create 'CustomRetriever' and 'corpus_retriever'
    #index_path = "C:/Users/sagie/PycharmProjects/LA_Project/data/indexes/retrieve_index.faiss"
    index_path = ""
    index_save_path = "C:/Users/sagie/PycharmProjects/LA_Project/data/indexes/retrieve_index.faiss"
    documents_index_path = ""
    #documents_index_path = "C:/Users/sagie/PycharmProjects/LA_Project/data/indexes/documents_index.pkl"
    documents_save_index_path = "C:/Users/sagie/PycharmProjects/LA_Project/data/indexes/documents_index.pkl"
    corpus_retriever = CorpusAndRetriever(model_name="intfloat/e5-large-v2", index_path=index_path,
                                          doc_index_path=documents_index_path, index_save_path=index_save_path,
                                          doc_save_index_path=documents_save_index_path)

    graph_rag = GraphRAG(corpus_retriever=corpus_retriever, model_path="meta-llama/Llama-3.2-1B-Instruct")

    # Create RAGManager instance
    rag_manager = RAGManager(graph_rag, input_mode="text", audio_model_name="openai/whisper-large-v3-turbo")

    # Example document paths (These should be actual paths to your documents)
    document_paths = ["C:/Users/sagie/PycharmProjects/LA_Project/data/corpus/CFR-2024-title2-vol1.pdf"]
    rag_manager.add_documents_to_corpus(document_paths)

    rag_manager.start_system()


if __name__ == "__main__":
    main()