"""
Retrieval-Augmented Generation (RAG) Module

This module provides the RAG capabilities for the AI assistant.
It uses ChromaDB for vector storage and sentence-transformers for embeddings.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    RAG_DEPENDENCIES_AVAILABLE = True
except ImportError:
    logger.warning("RAG dependencies (chromadb, sentence-transformers) are not installed. RAG functionality will be disabled.")
    logger.info("To enable RAG, please install the required packages: pip install chromadb sentence-transformers")
    RAG_DEPENDENCIES_AVAILABLE = False
    # Define dummy classes for type hinting and to avoid further errors
    class SentenceTransformer:
        def __init__(self, model_name): pass
    class chromadb:
        class PersistentClient:
            def __init__(self, path): pass
            def get_or_create_collection(self, name): pass


class RAG:
    """A RAG system for storing and retrieving knowledge."""

    def __init__(self, persist_directory: str = "./rag_db", collection_name: str = "knowledge", model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the RAG system.
        """
        self.is_available = RAG_DEPENDENCIES_AVAILABLE
        if not self.is_available:
            self.embedding_model = None
            self.client = None
            self.collection = None
            return

        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")

            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"ChromaDB collection '{collection_name}' loaded/created from {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            self.is_available = False


    def _chunk_text(self, text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
        """
        Split a text into overlapping chunks.

        Args:
            text: The text to split.
            chunk_size: The size of each chunk.
            chunk_overlap: The overlap between chunks.

        Returns:
            A list of text chunks.
        """
        words = re.split(r'\s+', text)
        chunks = []
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    def add_document(self, document_path: str) -> str:
        """
        Add a document to the knowledge base.

        Args:
            document_path: The path to the document to add.

        Returns:
            A status message.
        """
        if not self.is_available:
            return "Error: RAG system is not available because dependencies are not installed."

        if not os.path.exists(document_path):
            return f"Error: Document not found at {document_path}"

        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return f"Error: Could not read the document. {e}"

        chunks = self._chunk_text(content)
        embeddings = self.embedding_model.encode(chunks).tolist()
        ids = [f"{os.path.basename(document_path)}_{i}" for i in range(len(chunks))]

        metadatas = [{"source": document_path} for _ in range(len(chunks))]

        try:
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            return f"Successfully added document {document_path} to the knowledge base."
        except Exception as e:
            logger.error(f"Error adding document to ChromaDB: {e}")
            return f"Error: Could not add the document to the knowledge base. {e}"

    def query(self, query_text: str, n_results: int = 3) -> list[str]:
        """
        Query the knowledge base for relevant information.

        Args:
            query_text: The query text.
            n_results: The number of results to return.

        Returns:
            A list of relevant document chunks.
        """
        if not self.is_available:
            logger.warning("RAG system not available for query.")
            return []

        try:
            query_embedding = self.embedding_model.encode([query_text]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            return results['documents'][0] if results and results['documents'] else []
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []
