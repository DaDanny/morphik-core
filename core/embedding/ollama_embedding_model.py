import logging
from typing import List, Union

import ollama

from core.config import get_settings
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk

logger = logging.getLogger(__name__)


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """
    Direct Ollama embedding model implementation that bypasses LiteLLM to avoid API bugs.
    Uses the ollama library directly for reliable embedding generation.
    """

    def __init__(self, model_key: str):
        """
        Initialize Ollama embedding model with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
        """
        settings = get_settings()
        self.model_key = model_key

        # Get the model configuration from registered_models
        if not hasattr(settings, "REGISTERED_MODELS") or model_key not in settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.REGISTERED_MODELS[model_key]
        self.dimensions = settings.VECTOR_DIMENSIONS
        
        # Extract Ollama-specific configuration
        self.api_base = self.model_config.get("api_base", "http://localhost:11434")
        self.model_name = self.model_config.get("model_name", "")
        
        # Extract the actual model name from litellm format (e.g., "ollama/nomic-embed-text" -> "nomic-embed-text")
        if "/" in self.model_name:
            self.ollama_model_name = self.model_name.split("/")[-1]
        else:
            self.ollama_model_name = self.model_name
            
        # Initialize Ollama client
        self.client = ollama.AsyncClient(host=self.api_base)
        
        logger.info(f"Initialized Ollama embedding model with model_key={model_key}, "
                   f"ollama_model={self.ollama_model_name}, api_base={self.api_base}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents using direct Ollama client.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors (one per document)
        """
        if not texts:
            return []

        try:
            embeddings = []
            
            # Process each text individually for reliability
            for text in texts:
                response = await self.client.embeddings(
                    model=self.ollama_model_name,
                    prompt=text
                )
                
                # Extract embedding from response
                if "embedding" in response:
                    embeddings.append(response["embedding"])
                else:
                    logger.error(f"No embedding found in Ollama response: {response}")
                    # Return zero vector as fallback
                    embeddings.append([0.0] * self.dimensions)

            # Validate dimensions
            if embeddings and len(embeddings[0]) != self.dimensions:
                logger.warning(
                    f"Embedding dimension mismatch: got {len(embeddings[0])}, expected {self.dimensions}. "
                    f"Please update your VECTOR_DIMENSIONS setting to match the actual dimension."
                )
                # Update dimensions to match actual
                self.dimensions = len(embeddings[0])

            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with direct Ollama client: {e}")
            raise

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query using direct Ollama client.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        result = await self.embed_documents([text])
        if not result:
            # In case of error, return zero vector
            return [0.0] * self.dimensions
        return result[0]

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[List[float]]:
        """
        Generate embeddings for chunks to be ingested into the vector store.

        Args:
            chunks: Single chunk or list of chunks to embed

        Returns:
            List of embedding vectors (one per chunk)
        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        texts = [chunk.content for chunk in chunks]
        
        # Batch embedding to respect token limits and improve performance
        settings = get_settings()
        batch_size = getattr(settings, "EMBEDDING_BATCH_SIZE", 50)  # Smaller batch for Ollama
        embeddings: List[List[float]] = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            logger.debug(f"Processing embedding batch {i//batch_size + 1} with {len(batch_texts)} texts")
            batch_embeddings = await self.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
            
        return embeddings

    async def embed_for_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return await self.embed_query(text) 
