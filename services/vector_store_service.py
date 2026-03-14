from typing import List, Dict, Any, Optional

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)
from loguru import logger
import uuid

from config import app_settings


# from config import app_settings


class VectorStoreService:
    def __init__(self):
        self.client = AsyncQdrantClient(host=app_settings.QDRANT_HOST, port=app_settings.QDRANT_PORT)
        self.collection_name = app_settings.QDRANT_COLLECTION_NAME
        self.vector_size = app_settings.VECTOR_SIZE

        logger.info(f"Connected to Qdrant at {app_settings.QDRANT_HOST}:{app_settings.QDRANT_PORT}")

    async def create_collection(self, recreate: bool = False):
        """Create or recreate the collection"""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_list = collections.collections
            collection_exists = any(col.name == self.collection_name for col in collection_list)

            if collection_exists and recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                await self.client.delete_collection(self.collection_name)
                collection_exists = False

            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    async def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Insert or update chunks in the vector store"""
        try:
            points = []

            for chunk in chunks:
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk['embedding'],
                    payload={
                        'text': chunk['text'],
                        'chunk_id': chunk['id'],
                        'metadata': chunk['metadata']
                    }
                )
                points.append(point)

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Upserted batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}")

            logger.info(f"Successfully upserted {len(points)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error upserting chunks: {e}")
            raise

    async def search(
            self,
            query_vector: np.ndarray,
            limit: int = 5,
            score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            search_result = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold
            )

            results = []
            for hit in search_result.points:

                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'text': hit.payload['text'],
                    'chunk_id': hit.payload['chunk_id'],
                    'metadata': hit.payload['metadata']
                })

            return results

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}