import asyncio
import sys
import os

from config import app_settings
from services.docment_processor_service import DocumentProcessorService
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

# app_settings = Settings()
async def process_n_store(pdf_path: str):

    # Initialize services
    logger.info("Initializing services...")

    processor = DocumentProcessorService()

    embedding_service = EmbeddingService()

    # Get embedding dimension
    vector_size = embedding_service.get_embedding_dimension()
    logger.info(f"Embedding dimension: {vector_size}")

    vector_store = VectorStoreService()

    # Create collection
    await vector_store.create_collection(recreate=True)

    # Process document
    logger.info(f"Processing PDF: {pdf_path}")
    chunks = processor.process_document(pdf_path)

    # Generate embeddings
    logger.info("Generating embeddings...")
    chunks_with_embeddings = embedding_service.embed_chunks(chunks)

    # Store in vector database
    logger.info("Storing in Qdrant...")
    await vector_store.upsert_chunks(chunks_with_embeddings)

    # Verify storage
    info = vector_store.get_collection_info()
    logger.info(f"Collection info: {info}")

    logger.info("✅ Document processing complete!")


if __name__ == "__main__":
    print("Usage: python process_n_store_service.py <path_to_pdf>")
    if len(sys.argv) != 2:
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    asyncio.run(process_n_store(pdf_path))
    # process_n_store(pdf_path)