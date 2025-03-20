import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from work.Abstract_Graph_RAG import CustomGraphRAG
from work.LLM_prompts import rounting_prompt, RAG_prompt, QA_prompt
from work.Abstract_Retriver_Corpus import CustomRetrieverAndCorpus


class GraphRAG(CustomGraphRAG):
    def __init__(self, corpus_retriever: CustomRetrieverAndCorpus, model_path="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__(corpus_retriever)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.llm, self.tokenizer = self.load_llm(model_path)

    def load_llm(self, model_path="meta-llama/Llama-3.2-1B-Instruct"):
        """
        Load the LLM model.
        :param model_path: Path to the model.
        :return: transformers LLM and tokenizer.
        """
        logging.info(f"Loading the model {model_path} and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        # Add PAD token (EOS token) to the tokenizer and model
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        model.config.pad_token_id = tokenizer.pad_token_id
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer

    def generate_response(self, prompt: str, max_new_tokens=512):
        """
        Generate a response using the loaded LLM model from a given prompt.
        :param prompt: Prompt to generate a response for.
        :param max_new_tokens: Maximum number of tokens to generate.
        :return: Generated response string.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.llm.generate(**inputs, num_return_sequences=1, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return response.strip().splitlines()[0]

    def activate_RAG_graph(self, query: str, k: int = 5):
        """
        Perform retrieval and then generate a response based on a policy.
        Decide whether to fall back to the base LLM or use RAG based on retrieval confidence.
        :param query: Query to generate a response for.
        :param k: Number of documents to retrieve.
        """
        # Step 1: Use the routing prompt to decide the approach
        decision = self.decide_corpus_or_LLM(query)

        if decision.lower() == "yes":
            # Step 2: Retrieve and use RAG approach if decision is 'corpus'
            logging.info("Generating response using RAG...")
            response = self.retrieve_and_generate(query, k)
        elif decision.lower() == "no":
            logging.info("Generating response using only the LLM model...")
            # Fallback to base LLM if decision is 'LLM'
            response = self.generate_response(QA_prompt.format(query))
        else:
            response = "Error in decision-making process. LLM response isn't match the routing rules."

        return response

    def decide_corpus_or_LLM(self, query: str):
        """
        Decide whether to use the corpus or LLM based on the query.
        :param query: Query to decide the approach for.
        :return: Decision string - 'corpus' or 'LLM' or other text (if the LLM didn't follow instructions).
        """
        logging.info("Check if the query should be answered using the corpus or LLM...")
        decision_prompt = rounting_prompt.format(query)
        decision = self.generate_response(decision_prompt, max_new_tokens=5).strip()

        return decision

    def retrieve_and_generate(self, query: str, k: int = 5):
        """
        Retrieve documents and generate a response based on the query.
        :param query: Query to generate a response for.
        :param k: Number of documents to retrieve.
        :return: Generated response string.
        """
        retrieved_chunks = self.corpus_retriever.retrieve(query, k)
        logging.info(f"Retrieved {len(retrieved_chunks)} chunks from the corpus...")
        # Concatenate retrieved chunks for context
        context = "\n".join(retrieved_chunks)

        print(f"context: {context}")

        rag_prompt = RAG_prompt.format(context, query)
        logging.info(f"generating response based on the retrieved documents...")
        response = self.generate_response(rag_prompt)

        return response
