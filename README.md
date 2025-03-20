# RAG Syste

## 📖 About This Project  
This project is an implementation of various concepts I have self-learned about **Retrieval-Augmented Generation (RAG)** and **audio processing**, particularly in the context of **Large Language Models (LLMs)**. 
Through this project, I explored key areas such as:
- **RAG-based information retrieval**, utilizing embedding models for document search.
- **Efficient indexing with FAISS**, to enable fast and accurate retrieval.
- **LLM-powered response generation**, using transformers to provide contextual answers.
- **Speech-to-text processing**, leveraging Whisper for seamless voice interactions.
  
---

## 🚀 Overview  
This **RAG (Retrieval-Augmented Generation) System** is a powerful AI-driven pipeline designed for **intelligent document retrieval, multimodal input processing (text & audio), and adaptive response generation**. The system leverages **graph-based retrieval, FAISS indexing, and deep learning models** to provide accurate and context-aware responses to user queries.

### 🔹 Key Capabilities
✅ **Advanced Document Retrieval**  
- Utilizes FAISS indexing for efficient similarity search.  
- Dynamically loads and indexes documents from a specified folder.  

✅ **Graph-Based RAG (GraphRAG)**  
- Uses structured **Graph Retrieval-Augmented Generation** to enhance responses with contextual understanding.  

✅ **Multimodal Input Handling**  
- Supports **text** and **audio-based queries**.  
- Integrates **OpenAI Whisper** for speech-to-text processing.  

✅ **Adaptive Routing**  
- Can route queries based on intent (e.g., direct retrieval vs. generative response), also called LLM-fallback.  
- Supports **customized workflows** based on query complexity.  

---

## 🛠️ Installation & Setup  

### Install Dependencies  
Ensure you have Python **3.8+** installed, then install the required dependencies:  
```sh
pip install -r requirements.txt
```

### Run the System
```sh
python path_to_main.py --json_file path_to_config.json
```
