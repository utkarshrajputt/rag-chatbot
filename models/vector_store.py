"""
Vector Store Module
Handles embedding generation and FAISS-based similarity search.
"""

import numpy as np
import faiss
import logging
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles text embedding generation using Sentence Transformers."""
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', device: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the SentenceTransformer model
            device (str): Device to run on ('cpu', 'cuda', or None for auto-detection)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading embedding model: {model_name} on {device}")
        try:
            # Get token from environment
            import os
            hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
            
            # Load model with token if available
            if hf_token:
                self.model = SentenceTransformer(model_name, device=device, token=hf_token)
            else:
                self.model = SentenceTransformer(model_name, device=device)
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not texts:
            logger.warning("Empty text list provided for embedding generation")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                convert_to_tensor=True,
                show_progress_bar=len(texts) > 10
            )
            
            # Convert to numpy array
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.generate_embeddings([text])[0]

class VectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim (int): Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
        self.chunks = []
        self.metadata = []
        
        logger.info(f"Initialized FAISS vector store with dimension {embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict[str, Any]] = None):
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings (np.ndarray): Embeddings to add
            chunks (List[str]): Corresponding text chunks
            metadata (List[Dict]): Optional metadata for each chunk
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings and chunks must match")
        
        # Ensure embeddings are in correct format
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in chunks])
        
        logger.info(f"Added {len(embeddings)} embeddings to vector store. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            k (int): Number of results to return
            
        Returns:
            Tuple: (distances, indices, chunks, metadata)
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return np.array([]), np.array([]), [], []
        
        # Ensure query embedding is in correct format
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Perform search
        k = min(k, self.index.ntotal)  # Ensure k doesn't exceed available vectors
        distances, indices = self.index.search(query_embedding, k)
        
        # Get corresponding chunks and metadata
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        retrieved_metadata = [self.metadata[i] for i in indices[0]]
        
        logger.info(f"Retrieved {len(retrieved_chunks)} similar chunks")
        return distances, indices, retrieved_chunks, retrieved_metadata
    
    def save(self, filepath: str):
        """Save the vector store to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save chunks and metadata
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'metadata': self.metadata,
                    'embedding_dim': self.embedding_dim
                }, f)
            
            logger.info(f"Vector store saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load(self, filepath: str):
        """Load the vector store from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load chunks and metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
                self.embedding_dim = data['embedding_dim']
            
            logger.info(f"Vector store loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def clear(self):
        """Clear the vector store."""
        self.index.reset()
        self.chunks = []
        self.metadata = []
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.embedding_dim,
            'total_chunks': len(self.chunks),
            'index_type': type(self.index).__name__
        }

class RAGVectorStore:
    """Complete RAG vector store combining embedding generation and search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """Initialize the RAG vector store."""
        self.embedding_generator = EmbeddingGenerator(model_name, device)
        self.vector_store = VectorStore(self.embedding_generator.embedding_dim)
        
    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector store."""
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        self.vector_store.add_embeddings(embeddings, chunks, metadata)
    
    def search_documents(self, query: str, k: int = 5) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for relevant documents.
        
        Returns:
            Tuple: (chunks, metadata, scores)
        """
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        distances, indices, chunks, metadata = self.vector_store.search(query_embedding, k)
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        scores = [float(1 / (1 + dist)) for dist in distances[0]] if len(distances) > 0 else []
        
        return chunks, metadata, scores
    
    def save(self, filepath: str):
        """Save the complete vector store."""
        self.vector_store.save(filepath)
    
    def load(self, filepath: str):
        """Load the complete vector store."""
        self.vector_store.load(filepath)
    
    def clear(self):
        """Clear the vector store."""
        self.vector_store.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = self.vector_store.get_stats()
        stats['model_name'] = self.embedding_generator.model_name
        stats['device'] = self.embedding_generator.device
        return stats
