from typing import List, Dict, Any
from pypdf import PdfReader
from loguru import logger
import re
from transformers import AutoTokenizer

from config import app_settings


class DocumentProcessorService:

    def __init__(self):
        self.chunk_size = app_settings.CHUNK_SIZE
        self.chunk_overlap = app_settings.CHUNK_OVERLAP
        # self.max_tokens = max_tokens

        # Initialize tokenizer
        try:
            self.encoding = AutoTokenizer.from_pretrained(app_settings.EMBEDDING_MODEL)
        except:
            logger.warning("Could not load AutoTokenizer, using character-based chunking")
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            return len(text) // 4  # Rough estimate

    def load_pdf(self, file_path: str) -> str:
        """Load and extract text from PDF"""
        try:
            reader = PdfReader(file_path)
            text = ""

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"

            logger.info(f"Loaded PDF with {len(reader.pages)} pages")
            return text

        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\"\'\/]', '', text)
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)

        return text.strip()

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:

        # Clean text first
        text = self.clean_text(text)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds max_tokens, split it further
            if sentence_tokens > self.chunk_size:
                logger.warning(f"Sentence exceeds max_tokens ({sentence_tokens}), splitting further")
                # Split by words
                words = sentence.split()
                word_chunk = []
                word_tokens = 0

                for word in words:
                    word_token_count = self.count_tokens(word)
                    if word_tokens + word_token_count > self.chunk_size - 20:
                        if word_chunk:
                            chunk_text = ' '.join(word_chunk)
                            chunks.append({
                                'id': f'chunk_{chunk_id}',
                                'text': chunk_text,
                                'metadata': {
                                    'chunk_id': chunk_id,
                                    'char_count': len(chunk_text),
                                    # 'token_count': self.count_tokens(chunk_text)
                                }
                            })
                            chunk_id += 1
                        word_chunk = [word]
                        word_tokens = word_token_count
                    else:
                        word_chunk.append(word)
                        word_tokens += word_token_count

                if word_chunk:
                    chunk_text = ' '.join(word_chunk)
                    chunks.append({
                        'id': f'chunk_{chunk_id}',
                        'text': chunk_text,
                        'metadata': {
                            'chunk_id': chunk_id,
                            'char_count': len(chunk_text),
                            # 'token_count': self.count_tokens(chunk_text)
                        }
                    })
                    chunk_id += 1
                continue

            # If adding this sentence exceeds limit, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'id': f'chunk_{chunk_id}',
                    'text': chunk_text,
                    'metadata': {
                        'chunk_id': chunk_id,
                        'char_count': len(chunk_text),
                        # 'token_count': current_tokens
                    }
                })

                # Create overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 1 else []
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
                chunk_id += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': chunk_text,
                'metadata': {
                    'chunk_id': chunk_id,
                    'char_count': len(chunk_text),
                    'token_count': self.count_tokens(chunk_text)
                }
            })

        logger.info(f"Created {len(chunks)} chunks from document")

        # Log token statistics
        token_counts = [chunk['metadata']['token_count'] for chunk in chunks]
        if token_counts:
            logger.info(
                f"Token stats - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts) / len(token_counts):.1f}")

        return chunks

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Complete document processing pipeline"""
        # Load PDF
        text = self.load_pdf(file_path)

        # Chunk text
        chunks = self.chunk_text(text)

        # Add source metadata to all chunks
        for chunk in chunks:
            chunk['metadata']['source'] = file_path
            chunk['metadata']['total_chunks'] = len(chunks)

        return chunks