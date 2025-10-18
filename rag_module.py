#!/usr/bin/env python3
"""
rag_module.py - Retrieval-Augmented Generation Module for EMOSIC AI
This module provides RAG functionality using local LLM (HuggingFacePipeline)
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import torch

class MusicRAG:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = self._setup_local_llm()
        self.vectorstore = None
        self.qa_chain = None
        self._build_knowledge_base()

    def _setup_local_llm(self):
        # EDIT: Change model_name to your preferred HuggingFace model (e.g., "meta-llama/Llama-2-7b-chat-hf" if you have access)
        # For gated models, you may need to login: huggingface-cli login
        model_name = "microsoft/DialoGPT-medium"  # Free conversational model, no token needed

        # EDIT: Adjust torch_dtype for your hardware (float16 for GPU, float32 for CPU)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Set to float32 for CPU compatibility
            device_map="cpu"  # Set to "cpu" for CPU-only systems
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,  # Reduced for faster response on CPU
            temperature=0.5  # Lower temperature for more focused responses
        )
        return HuggingFacePipeline(pipeline=pipe)

    def _build_knowledge_base(self):
        """Index song data and rules as documents"""
        documents = []

        # Add song metadata as documents
        for _, row in self.dataset.iterrows():
            text = f"Song: {row['title']} by {row['artist']}. Tempo: {row['tempo']}, Valence: {row['valence']}, Energy: {row['energy']}, Key: {row['key']}. Emotion: {self._infer_emotion(row)}"
            documents.append(Document(page_content=text, metadata={"type": "song", "id": row.name}))

        # Add rules as documents
        from rules import SIMPLIFIED_RULES  # EDIT: Ensure rules.py is in the same directory
        for rule in SIMPLIFIED_RULES:
            text = f"Rule {rule['id']}: {rule['rule']}. Conclusion: {rule['conclusion']}. Confidence: {rule['confidence']}"
            documents.append(Document(page_content=text, metadata={"type": "rule", "id": rule['id']}))

        # Split and store
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(docs, self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vectorstore.as_retriever())

    def _infer_emotion(self, song_row):
        """Simple emotion inference (reuse from your system)"""
        from rules import fuzzy_controller  # EDIT: Ensure rules.py is imported
        emotion, _ = fuzzy_controller.infer_from_song(song_row)
        return emotion

    def augment_explanation(self, base_explanation: str, query: str) -> str:
        """Augment existing explanations with RAG insights"""
        rag_response = self.qa_chain.run(f"Based on this: {base_explanation}. Answer: {query}")
        return f"{base_explanation}\n\nRAG Insight: {rag_response}"

    def standalone_query(self, question: str) -> str:
        """Standalone RAG query for new features"""
        return self.qa_chain.run(question)
