"""
RAG (Retrieval-Augmented Generation) setup using ChromaDB.
Manages vector stores for drug interactions, clinical guidelines, and patient education.
"""

from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_embeddings, settings
import os


class RAGManager:
    """Manages ChromaDB collections for RAG."""

    def __init__(self):
        """Initialize RAG manager with ChromaDB collections."""
        self.embeddings = get_embeddings()

        # Ensure chroma_db directory exists
        os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Initialize vector stores
        self._init_vector_stores()

    def _init_vector_stores(self):
        """Initialize the three ChromaDB collections."""

        # Drug Interactions Collection
        self.drug_store = Chroma(
            client=self.client,
            collection_name="drug_interactions",
            embedding_function=self.embeddings,
        )

        # Clinical Guidelines Collection
        self.guidelines_store = Chroma(
            client=self.client,
            collection_name="clinical_guidelines",
            embedding_function=self.embeddings,
        )

        # Patient Education Collection
        self.education_store = Chroma(
            client=self.client,
            collection_name="patient_education",
            embedding_function=self.embeddings,
        )

    def get_drug_retriever(self, k: int = 5, score_threshold: float = 0.75):
        """
        Get retriever for drug interaction database.

        Args:
            k: Number of results to retrieve
            score_threshold: Minimum similarity score

        Returns:
            LangChain retriever instance
        """
        return self.drug_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold}
        )

    def get_guidelines_retriever(self, k: int = 4, fetch_k: int = 10):
        """
        Get retriever for clinical guidelines with MMR (diversity).

        Args:
            k: Number of final results
            fetch_k: Number of candidates to fetch before MMR

        Returns:
            LangChain retriever instance
        """
        return self.guidelines_store.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={"k": k, "fetch_k": fetch_k}
        )

    def get_education_retriever(self, k: int = 3):
        """
        Get retriever for patient education materials.

        Args:
            k: Number of results to retrieve

        Returns:
            LangChain retriever instance
        """
        return self.education_store.as_retriever(
            search_kwargs={"k": k}
        )

    def add_documents(
        self,
        collection_name: str,
        documents: list[Document]
    ):
        """
        Add documents to a specific collection.

        Args:
            collection_name: Name of the collection ('drug_interactions', 'clinical_guidelines', 'patient_education')
            documents: List of LangChain Document objects
        """
        store_map = {
            "drug_interactions": self.drug_store,
            "clinical_guidelines": self.guidelines_store,
            "patient_education": self.education_store,
        }

        if collection_name not in store_map:
            raise ValueError(f"Invalid collection name: {collection_name}")

        store = store_map[collection_name]
        store.add_documents(documents)

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> list[Document]:
        """
        Chunk text into Documents suitable for RAG.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
            metadata: Metadata to attach to all chunks

        Returns:
            List of Document objects
        """
        chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)

        return [
            Document(
                page_content=chunk,
                metadata=metadata or {}
            )
            for chunk in chunks
        ]

    def search_drug_interactions(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for drug interaction information.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of dicts with 'content', 'source', 'score'
        """
        retriever = self.get_drug_retriever(k=k)
        docs = retriever.invoke(query)

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": doc.metadata.get("score", 0.0),
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    def search_clinical_guidelines(self, query: str, k: int = 4) -> list[dict]:
        """
        Search clinical guidelines database.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of dicts with 'content', 'source', 'score'
        """
        retriever = self.get_guidelines_retriever(k=k)
        docs = retriever.invoke(query)

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": doc.metadata.get("score", 0.0),
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    def reset_collection(self, collection_name: str):
        """
        Clear all documents from a collection.

        Args:
            collection_name: Name of collection to reset
        """
        try:
            self.client.delete_collection(name=collection_name)
            self._init_vector_stores()
        except Exception as e:
            print(f"Error resetting collection {collection_name}: {e}")

    def get_collection_count(self, collection_name: str) -> int:
        """
        Get number of documents in a collection.

        Args:
            collection_name: Name of collection

        Returns:
            Number of documents
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection.count()
        except Exception:
            return 0

    def query_education(self, query: str, k: int = 3) -> list[dict]:
        """
        Query patient education materials.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of dicts with 'content', 'source', 'score'
        """
        retriever = self.get_education_retriever(k=k)
        docs = retriever.invoke(query)

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": doc.metadata.get("score", 0.0),
                "metadata": doc.metadata
            }
            for doc in docs
        ]


# Global RAG manager instance
rag_manager = RAGManager()
