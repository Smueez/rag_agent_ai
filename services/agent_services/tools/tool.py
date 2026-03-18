from agents import function_tool

from config import app_settings
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService


class Tool:
    @staticmethod
    @function_tool
    async def fetch_data(query: str):
        """
        Search the vector database for relevant context chunks.

        Args:
            query: the rewritten query to search with

        Returns:
            A list of chunk objects with text and metadata
        """
        vector_service = VectorStoreService()
        embedding_service = EmbeddingService()
        query_vector = embedding_service.generate_embedding(query)
        result = await vector_service.search(
            query_vector=query_vector,
            limit=app_settings.TOP_K_RESULTS,
            score_threshold=app_settings.SIMILARITY_THRESHOLD
        )

        return result