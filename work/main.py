import argparse
import logging
import json
import os
import sys

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

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run agent with a JSON config file.")
    parser.add_argument("--json_file", type=str, help="Path to the configuration JSON file.", required=False)

    # Parse arguments
    args = parser.parse_args()

    # Try to get the config file from command-line argument first, then fallback to environment variable
    CONFIG_FILE_PATH = args.json_file
    logging.info(f"Loading configuration from {CONFIG_FILE_PATH}")

    # Load configuration from the JSON file
    with open(CONFIG_FILE_PATH, "r") as f:
        config = json.load(f)

    # Extract paths and settings from the config file
    index_path = config.get("index_path", "")
    index_save_path = config.get("index_save_path", "")
    documents_index_path = config.get("documents_index_path", "")
    documents_save_index_path = config.get("documents_save_index_path", "")

    # Extract model settings
    retriever_model_name = config.get("retriever_model_name", "intfloat/e5-large-v2")
    rag_model_path = config.get("rag_model_path", "meta-llama/Llama-3.2-1B-Instruct")
    input_mode = config.get("input_mode", "text")
    audio_model_name = config.get("audio_model_name", "openai/whisper-large-v3-turbo")
    if input_mode not in ["text", "audio"]:
        raise ValueError("Error: The input_mode must be either 'text' or 'audio'.")

    # Extract document folder path
    documents_path_folder = config.get("documents_path_folder", None)
    document_paths = None
    if documents_path_folder is not None and not os.path.isdir(documents_path_folder):
        raise ValueError(
            f"Error: The specified documents_path_folder '{documents_path_folder}' is not a valid directory.")
    elif documents_path_folder is not None:
        # Collect all document file paths in the folder
        document_paths = [os.path.join(documents_path_folder, f) for f in os.listdir(documents_path_folder) if
                          os.path.isfile(os.path.join(documents_path_folder, f))]
        logging.info(f"Found {len(document_paths)} documents in the folder to add to RAG.")
    # Initialize Corpus Retriever
    corpus_retriever = CorpusAndRetriever(
        model_name=retriever_model_name,
        index_path=index_path,
        doc_index_path=documents_index_path,
        index_save_path=index_save_path,
        doc_save_index_path=documents_save_index_path
    )

    # Initialize GraphRAG
    graph_rag = GraphRAG(corpus_retriever=corpus_retriever, model_path=rag_model_path)

    # Initialize RAGManager
    rag_manager = RAGManager(graph_rag, input_mode=input_mode, audio_model_name=audio_model_name)

    # Add all documents from the folder to the corpus
    if document_paths is not None:
        rag_manager.add_documents_to_corpus(document_paths)

    logging.info(f"Starting system...")
    rag_manager.start_system()


if __name__ == "__main__":
    main()