"""
Text Chunking Module
Implements recursive character splitting with overlap for optimal semantic coherence.
"""

import nltk
import re
import logging
from typing import List, Optional
from dataclasses import dataclass

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 600  # characters
    overlap_size: int = 75  # characters
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", "! ", "? ", " "]

class TextChunker:
    """Implements recursive character splitting with semantic awareness."""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        logger.info(f"TextChunker initialized with chunk_size={self.config.chunk_size}, overlap={self.config.overlap_size}")
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by a specific separator."""
        if separator == " ":
            return text.split(separator)
        return text.split(separator)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits that are too small."""
        merged = []
        current = ""
        
        for split in splits:
            if not split.strip():
                continue
                
            test_chunk = current + separator + split if current else split
            
            if len(test_chunk) <= self.config.chunk_size:
                current = test_chunk
            else:
                if current:
                    merged.append(current.strip())
                current = split
        
        if current:
            merged.append(current.strip())
        
        return [chunk for chunk in merged if chunk.strip()]
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using different separators."""
        if not separators:
            # Base case: split by characters if no separators left
            return [text[i:i+self.config.chunk_size] for i in range(0, len(text), self.config.chunk_size)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = self._split_by_separator(text, separator)
        
        if len(splits) == 1:
            # Separator didn't split the text, try next separator
            return self._recursive_split(text, remaining_separators)
        
        # Merge small splits
        merged_splits = self._merge_splits(splits, separator)
        
        final_chunks = []
        for split in merged_splits:
            if len(split) <= self.config.chunk_size:
                final_chunks.append(split)
            else:
                # Split is still too large, recursively split with remaining separators
                sub_chunks = self._recursive_split(split, remaining_separators)
                final_chunks.extend(sub_chunks)
        
        return final_chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.config.overlap_size == 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            
            # Get overlap from previous chunk
            prev_chunk = chunks[i-1]
            overlap = prev_chunk[-self.config.overlap_size:] if len(prev_chunk) > self.config.overlap_size else prev_chunk
            
            # Add overlap to current chunk
            overlapped_chunk = overlap + " " + chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive character splitting with overlap.
        
        Args:
            text (str): Input text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        text = text.strip()
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.config.chunk_size:
            return [text]
        
        # Perform recursive splitting
        chunks = self._recursive_split(text, self.config.separators)
        
        # Add overlap between chunks
        overlapped_chunks = self._add_overlap(chunks)
        
        # Filter out empty chunks and chunks that are too small
        final_chunks = [
            chunk.strip() for chunk in overlapped_chunks 
            if chunk.strip() and len(chunk.strip()) >= 50  # Minimum chunk size
        ]
        
        logger.info(f"Text chunked into {len(final_chunks)} chunks")
        return final_chunks
    
    def chunk_with_metadata(self, text: str, source_file: str = "") -> List[dict]:
        """
        Chunk text and return with metadata.
        
        Args:
            text (str): Input text to chunk
            source_file (str): Source file name
            
        Returns:
            List[dict]: List of chunks with metadata
        """
        chunks = self.chunk_text(text)
        
        return [
            {
                "content": chunk,
                "chunk_id": i,
                "source_file": source_file,
                "char_count": len(chunk),
                "word_count": len(chunk.split())
            }
            for i, chunk in enumerate(chunks)
        ]

class SentenceChunker:
    """Alternative chunker that splits by sentences with token limits."""
    
    def __init__(self, max_tokens: int = 200):
        self.max_tokens = max_tokens
        logger.info(f"SentenceChunker initialized with max_tokens={max_tokens}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text by sentences with token limits.
        
        Args:
            text (str): Input text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or not text.strip():
            return []
        
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                # Current chunk is full, start new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        logger.info(f"Text chunked into {len(chunks)} sentence-based chunks")
        return [chunk for chunk in chunks if chunk.strip()]
