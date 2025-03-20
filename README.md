# RAG Syste

## 📖 About This Project  
As part of my **self-learning journey in Data Science**, I explored **audio processing with LLMs** and **Retrieval-Augmented Generation (RAG)**. This project is a practical implementation of the concepts I learned, combining **document retrieval, speech-to-text processing, and AI-powered knowledge generation**. By integrating **FAISS-based search, GraphRAG, and OpenAI Whisper for audio input**, I built a system capable of handling **both textual and spoken queries** with intelligent response generation.

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
