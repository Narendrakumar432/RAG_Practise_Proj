-- Enable the PostgreSQL vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the document_chunks table to store text chunks, page numbers, and 384-dimensional embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    embedding VECTOR(384)
);

-- Index for vector similarity search using cosine distance
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops);
