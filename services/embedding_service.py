import traceback
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from config import app_settings
from services.embedding_model_service import ModelService
from utils.singelton_utils import singleton


# import tiktoken

@singleton
class EmbeddingService:
    def __init__(self):
        try:
            self.deployment_name = app_settings.EMBEDDING_MODEL
            self.max_tokens = app_settings.MAX_TOKEN
            self.model_service = ModelService()
            self.model : Optional[SentenceTransformer] = None
            try:
                self.model = self.model_service.get_Sentence_trancsformer_model()
            except Exception as e:
                traceback.print_exc()
                logger.error("Could not load embedding model, using character-based estimation")
                logger.error("e")


            logger.info(f"Initialized EmbeddingService with deployment: {self.deployment_name}")

        except Exception as e:
            logger.error(f"Error initializing AzureOpenAI client: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text with retry logic"""
        try:

            embeddings = self.model.encode(
                text,
                batch_size=app_settings.BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(
            self,
            texts: List[str]
    ) ->np.ndarray:
        """Generate embeddings for multiple texts in batches"""

        try:
            # Verify batch doesn't exceed limits

            embeddings = self.model.encode(
                texts,
                batch_size=app_settings.BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )


            logger.info(
                f"Generated embeddings for {len(texts)}")
            return embeddings
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in batch {e}")
        return np.ndarray([])

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embeddings to document chunks"""
        texts = [chunk['text'] for chunk in chunks]


        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings_batch(texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding

        logger.info("Successfully embedded all chunks")
        return chunks

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)