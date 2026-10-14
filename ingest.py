import os
import sys

# Silence TensorFlow / symlink warnings before loading heavy libraries
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import re
import argparse
from typing import List, Tuple
from dotenv import load_dotenv
import pymupdf  # PyMuPDF
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DATABASE_URL = os.getenv("DATABASE_URL")

DEFAULT_CHUNK_SIZE = 500      # characters (~90-110 words)
DEFAULT_CHUNK_OVERLAP = 100   # characters (~18-22 words, ~20% overlap)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_db_connection():
    """Establish and return a PostgreSQL connection."""
    if DATABASE_URL:
        conn = psycopg2.connect(DATABASE_URL)
    else:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
    register_vector(conn)
    return conn


def ensure_table_exists(conn):
    """Ensure the pgvector extension and document_chunks table exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                embedding VECTOR(384)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
    conn.commit()


def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text page-by-page from the PDF using PyMuPDF.
    Returns a list of (page_number, text) tuples with 1-indexed page numbers.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    pages_data = []

    for page_idx, page in enumerate(doc, start=1):
        raw_text = page.get_text("text").strip()
        # Clean excessive newlines/spaces while preserving paragraphs
        cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text).strip()
        if cleaned_text:
            pages_data.append((page_idx, cleaned_text))

    doc.close()
    return pages_data


def chunk_text(
    pages_data: List[Tuple[int, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Tuple[str, int]]:
    """
    Split text into overlapping chunks while preserving original page numbers.
    Uses word-boundary snapping to avoid splitting in the middle of words.
    Returns a list of (chunk_text, page_number) tuples.
    """
    chunks = []

    for page_number, page_text in pages_data:
        text_len = len(page_text)
        if text_len <= chunk_size:
            chunks.append((page_text, page_number))
            continue

        start = 0
        while start < text_len:
            end = min(start + chunk_size, text_len)

            # Snap to nearest whitespace or newline within the window to prevent mid-word cutting
            if end < text_len:
                boundary_search_start = start + chunk_size // 2
                last_space = page_text.rfind(" ", boundary_search_start, end)
                last_newline = page_text.rfind("\n", boundary_search_start, end)
                boundary = max(last_space, last_newline)
                if boundary > start:
                    end = boundary

            chunk = page_text[start:end].strip()
            if chunk:
                chunks.append((chunk, page_number))

            if end >= text_len:
                break

            # Step forward with overlap
            step = chunk_size - chunk_overlap
            next_start = end - chunk_overlap
            if next_start <= start:
                next_start = start + max(1, step)

            # Avoid trailing duplicate micro-chunks that are already subsumed by overlap
            if text_len - next_start <= chunk_overlap:
                break

            start = next_start

    return chunks


def ingest_pdf(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset_db: bool = False,
):
    """
    Complete ingestion pipeline:
    1. Extracts text preserving page numbers.
    2. Chunks text with overlap.
    3. Generates 384-dimensional embeddings using all-MiniLM-L6-v2.
    4. Inserts chunks and vectors into PostgreSQL using pgvector.
    """
    print(f"=== Starting Ingestion for: {pdf_path} ===")
    
    # 1. Connect to PostgreSQL & ensure schema
    conn = get_db_connection()
    ensure_table_exists(conn)

    if reset_db:
        print("Clearing existing records in document_chunks...")
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")
        conn.commit()

    # 2. Extract text from PDF
    print("Extracting text from PDF...")
    pages_data = extract_text_from_pdf(pdf_path)
    total_pages = len(pages_data)
    print(f"Extracted {total_pages} page(s) with text.")

    if not pages_data:
        print("Warning: No text could be extracted from the document.")
        conn.close()
        return

    # 3. Chunk text
    print(f"Chunking text (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    chunks = chunk_text(pages_data, chunk_size, chunk_overlap)
    print(f"Generated {len(chunks)} overlapping chunk(s).")

    # 4. Generate embeddings
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunk_texts = [c[0] for c in chunks]
    print(f"Generating 384-dimensional embeddings for {len(chunk_texts)} chunk(s)...")
    embeddings = model.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=False)

    # 5. Store in PostgreSQL
    print("Inserting chunks and embeddings into PostgreSQL (pgvector)...")
    records = [
        (chunk_text, page_number, embedding.tolist())
        for (chunk_text, page_number), embedding in zip(chunks, embeddings)
    ]

    insert_sql = """
        INSERT INTO document_chunks (chunk_text, page_number, embedding)
        VALUES (%s, %s, %s);
    """
    with conn.cursor() as cur:
        execute_batch(cur, insert_sql, records, page_size=100)
    conn.commit()
    conn.close()

    print(f"Successfully ingested {len(records)} chunk(s) from {pdf_path} into PostgreSQL.")


def main():
    parser = argparse.ArgumentParser(
        description="PDF Vectorisation Pipeline - Ingest PDF into PostgreSQL pgvector"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the PDF file to ingest",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate the document_chunks table before ingestion",
    )

    args = parser.parse_args()
    ingest_pdf(
        pdf_path=args.file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset_db=args.reset,
    )


if __name__ == "__main__":
    main()
