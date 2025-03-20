import logging

import numpy as np
import speech_recognition as sr
import torch
import librosa
import soundfile as sf
import io
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, AutoModelForSpeechSeq2Seq, \
    AutoProcessor
from work.Abstract_Graph_RAG import CustomGraphRAG


class RAGManager:
    def __init__(self, graph_RAG: CustomGraphRAG, input_mode="text", audio_model_name: str = ""):
        """
        Initialize the RAGManager with a GraphRAG instance.

        :param graph_rag: An instance of the GraphRAG class.
        :param input_mode: "text" (default) for text input or "audio" for live speech-to-text queries.
        """
        self.graph_RAG = graph_RAG
        self.input_mode = input_mode.lower()
        logging.info(f"RAGManager initialized with GraphRAG instance in {self.input_mode} input mode.")

        if self.input_mode == "audio":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.load_audio_model(audio_model_name)

    def load_audio_model(self, model_path: str = "openai/whisper-large-v3-turbo"):
        """
        Load the OpenAI Whisper model for speech-to-text.
        :param model_path: The path to the model to load.
        :return: The loaded model.
        """
        logging.info("Loading OpenAI Whisper model for speech-to-text...")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        self.model = self.model.eval()
        self.model.config.forced_decoder_ids = None
        """
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device=self.device,
        )
        """
        logging.info("Whisper model loaded successfully.")

    def add_documents_to_corpus(self, file_path_list):
        """
        Add documents to the corpus using the DataProcessor in GraphRAG.

        :param file_path_list: A list of file paths to process and add to the corpus.
        """
        logging.info(f"Adding documents to the corpus from paths: {file_path_list}")
        self.graph_RAG.corpus_retriever.add_documents(file_path_list)
        logging.info("Documents successfully added to the corpus.")

    def handle_query(self, query, k=5):
        """
        Handle a query by sending it to the GraphRAG to activate the RAG mechanism.

        :param query: The query string to process.
        :param k: The number of documents to retrieve.
        :return: The generated response from the GraphRAG.
        """
        logging.info(f"Generate response to query: {query}")
        response = self.graph_RAG.activate_RAG_graph(query, k)
        logging.info("Query processed and response generated.")
        return response

    def start_system(self):
        """
        Starts the query processing system based on the input mode (text or audio).
        """
        if self.input_mode == "text":
            while True:
                query = input("Enter your query (type 'exit' to stop): ")
                if query.lower() == 'exit':
                    print("Exiting the query loop.")
                    break
                print(f"Response: {self.handle_query(query)}")

        elif self.input_mode == "audio":
            print("🎤 Audio mode activated. Speak to ask a query. Say 'exit' to stop.")
            while True:
                query = self.transcribe_audio()
                print(f"Query: {query}")
                if query is None:
                    print("No questions received. Exiting the audio query loop.")
                    break
                elif query and "exit" in query.lower().split():
                    print("Exiting the audio query loop.")
                    break
                elif query:
                    print(f"Response: {self.handle_query(query)}")

        logging.info("Exiting system...")

    def transcribe_audio(self):
        """
        Uses Whisper AI to listen to microphone input and convert speech to text.
        """
        generate_kwargs = {
            "max_new_tokens": 100,
            "temperature": (0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": False,
            "language": "english",
        }

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening for query... Speak now.")
            recognizer.adjust_for_ambient_noise(source)
            # Set a timeout and phrase time limit
            try:
                audio = recognizer.listen(source, timeout=30)
            except sr.WaitTimeoutError:
                print("No speech detected, exiting system.")
                return None

        try:
            # Convert audio to text using Whisper
            logging.info("Transcribing audio...")
            audio_bytes = io.BytesIO(audio.get_wav_data())
            #audio_bytes = audio.get_wav_data()

            # Convert WAV bytes to NumPy array (16kHz sample rate)
            audio_data, _ = librosa.load(audio_bytes, sr=16000, dtype=np.float32)
            #audio_data, _ = sf.read(audio_bytes, dtype="float32")
            #data_s16 = np.frombuffer(audio_bytes, dtype=np.int16, count=len(audio_bytes) // 2, offset=0)
            #audio_data = data_s16.astype(np.float32, order='C') / 32768.0

            inputs = self.processor(audio_data, return_tensors="pt", sampling_rate=16000)
            inputs["attention_mask"] = torch.ones_like(inputs["input_features"])
            inputs = inputs.to(self.device, dtype=torch.float16)
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs.input_features, attention_mask=inputs["attention_mask"])

            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True, generate_kwargs=generate_kwargs)[0]
            #transcription = self.pipe(audio_data, generate_kwargs=generate_kwargs)[0]["text"]

            logging.info(f"Transcription: {transcription}")
            return transcription
        except Exception as e:
            logging.error(f"Error in audio transcription: {e}")
            return None
