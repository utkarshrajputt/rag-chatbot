"""
DeepSeek R1 LLM Client
Handles communication with DeepSeek R1 model via OpenRouter API.
"""

import openai
import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM generation."""
    model: str = "deepseek/deepseek-r1"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    max_context_tokens: int = 8000

class DeepSeekClient:
    """Client for DeepSeek R1 model via OpenRouter API."""
    
    def __init__(self, api_key: str, config: Optional[LLMConfig] = None):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key (str): OpenRouter API key
            config (LLMConfig): Configuration for the model
        """
        self.api_key = api_key
        self.config = config or LLMConfig()
        
        # Configure OpenAI client for OpenRouter
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        logger.info(f"DeepSeek client initialized with model: {self.config.model}")
    
    def _prepare_context(self, retrieved_chunks: List[str]) -> str:
        """
        Prepare context from retrieved chunks with token limit consideration.
        
        Args:
            retrieved_chunks (List[str]): List of retrieved text chunks
            
        Returns:
            str: Prepared context string
        """
        if not retrieved_chunks:
            return ""
        
        context = "\n\n".join(retrieved_chunks)
        
        # Simple token estimation (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(context) // 4
        
        if estimated_tokens > self.config.max_context_tokens:
            # Truncate context to fit within token limit
            max_chars = self.config.max_context_tokens * 4
            context = context[:max_chars]
            logger.warning(f"Context truncated to {max_chars} characters due to token limit")
        
        return context
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create a RAG prompt for DeepSeek R1.
        
        Args:
            query (str): User query
            context (str): Retrieved context
            
        Returns:
            str: Formatted prompt
        """
        return f"""You are a helpful AI assistant. Answer the user's question using only the information provided in the context below. Be specific and accurate.

If the answer cannot be found in the context, clearly state "I don't have enough information in the provided context to answer this question."

Context:
{context}

Question: {query}

Answer:"""
    
    def generate_answer(
        self, 
        query: str, 
        retrieved_chunks: List[str],
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer using DeepSeek R1 model.
        
        Args:
            query (str): User query
            retrieved_chunks (List[str]): Retrieved context chunks
            custom_prompt (str): Optional custom prompt template
            
        Returns:
            Dict[str, Any]: Response containing answer and metadata
        """
        try:
            # Prepare context
            context = self._prepare_context(retrieved_chunks)
            
            # Create prompt
            if custom_prompt:
                prompt = custom_prompt.format(context=context, query=query)
            else:
                prompt = self._create_rag_prompt(query, context)
            
            logger.info(f"Generating answer for query: {query[:50]}...")
            
            # Call DeepSeek R1 via OpenRouter
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Prepare response metadata
            usage = response.usage
            response_data = {
                "answer": answer,
                "model": self.config.model,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0
                },
                "chunks_used": len(retrieved_chunks),
                "context_length": len(context)
            }
            
            logger.info(f"Answer generated successfully. Tokens used: {response_data['usage']['total_tokens']}")
            return response_data
            
        except openai.APIError as e:
            logger.error(f"OpenRouter API error: {e}")
            return {
                "answer": "I apologize, but I encountered an error while generating the answer. Please try again.",
                "error": str(e),
                "model": self.config.model
            }
        except Exception as e:
            logger.error(f"Unexpected error in answer generation: {e}")
            return {
                "answer": "I apologize, but I encountered an unexpected error. Please try again.",
                "error": str(e),
                "model": self.config.model
            }
    
    def generate_simple_answer(self, query: str) -> str:
        """
        Generate a simple answer without RAG context (for testing).
        
        Args:
            query (str): User query
            
        Returns:
            str: Generated answer
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error in simple answer generation: {e}")
            return f"Error generating answer: {str(e)}"
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to OpenRouter API.
        
        Returns:
            Dict[str, Any]: Test results
        """
        try:
            response = self.generate_simple_answer("Hello, can you hear me?")
            return {
                "success": True,
                "message": "Connection successful",
                "test_response": response
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection failed: {str(e)}",
                "test_response": None
            }
    
    def update_config(self, **kwargs):
        """Update model configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

class RAGPipeline:
    """Complete RAG pipeline with DeepSeek R1 integration."""
    
    def __init__(self, deepseek_client: DeepSeekClient, vector_store):
        """
        Initialize RAG pipeline.
        
        Args:
            deepseek_client (DeepSeekClient): DeepSeek R1 client
            vector_store: Vector store for document retrieval
        """
        self.llm_client = deepseek_client
        self.vector_store = vector_store
        logger.info("RAG pipeline initialized")
    
    def process_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a complete RAG query.
        
        Args:
            query (str): User query
            k (int): Number of chunks to retrieve
            
        Returns:
            Dict[str, Any]: Complete response with answer and metadata
        """
        logger.info(f"Processing RAG query: {query[:50]}...")
        
        # Retrieve relevant chunks
        chunks, metadata, scores = self.vector_store.search_documents(query, k)
        
        if not chunks:
            return {
                "answer": "I don't have any relevant information to answer your question.",
                "chunks": [],
                "metadata": [],
                "scores": [],
                "model": self.llm_client.config.model
            }
        
        # Generate answer
        response = self.llm_client.generate_answer(query, chunks)
        
        # Add retrieval metadata
        response.update({
            "chunks": chunks,
            "metadata": metadata,
            "scores": scores,
            "retrieval_k": k
        })
        
        return response
