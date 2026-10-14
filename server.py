import os
import sys

# Silence TensorFlow / symlink warnings before loading heavy libraries
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from contextlib import asynccontextmanager
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# Load environment configuration
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DATABASE_URL = os.getenv("DATABASE_URL")

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global references for connection pool and embedding model
db_pool: Optional[pool.ThreadedConnectionPool] = None
embedding_model: Optional[SentenceTransformer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initialize embedding model and database pool on startup."""
    global db_pool, embedding_model

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded successfully.")

    print("Initializing PostgreSQL connection pool...")
    if DATABASE_URL:
        db_pool = pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)
    else:
        db_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
    print("PostgreSQL connection pool initialized.")

    yield

    if db_pool:
        db_pool.closeall()
        print("PostgreSQL connection pool closed.")


app = FastAPI(
    title="Nestack PDF Vectorisation Retrieval API",
    description="Vector similarity retrieval API powered by FastAPI, Sentence Transformers, and PostgreSQL pgvector.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend or Postman access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response schemas
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        example="How does the ingestion process work?",
        description="Search query string",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=50,
        example=3,
        description="Number of top matching chunks to retrieve",
    )


class ChunkResult(BaseModel):
    chunk_text: str = Field(..., description="Relevant text extracted from the PDF")
    page_number: int = Field(..., description="Original PDF page number (1-indexed)")
    score: float = Field(..., description="Cosine similarity score (1 - cosine distance)")


@app.get("/", tags=["Health"])
def root():
    """Root endpoint providing service metadata."""
    return {
        "service": "Nestack PDF Vectorisation Retrieval API",
        "model": EMBEDDING_MODEL_NAME,
        "vector_dimension": 384,
        "vector_db": "PostgreSQL + pgvector",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check verifying database connection and chunk inventory."""
    if not db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool is not initialized.",
        )

    conn = db_pool.getconn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            total_chunks = cur.fetchone()[0]
        return {
            "status": "healthy",
            "model_loaded": embedding_model is not None,
            "total_chunks_in_db": total_chunks,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database health check failed: {str(e)}",
        )
    finally:
        db_pool.putconn(conn)


@app.post("/query", response_model=List[ChunkResult], tags=["Retrieval"])
def query_chunks(payload: QueryRequest):
    """
    Retrieve semantically similar chunks for a given query:
    1. Validates the query and top_k parameter.
    2. Encodes the query text into a 384-dimensional vector using all-MiniLM-L6-v2.
    3. Runs cosine distance similarity search in PostgreSQL via pgvector (<=>).
    4. Returns array of { chunk_text, page_number, score }.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    if embedding_model is None or db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server components are not ready.",
        )

    # 1. Generate query embedding
    try:
        query_vector = embedding_model.encode(
            payload.query.strip(), normalize_embeddings=False
        ).tolist()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating query embedding: {str(e)}",
        )

    # 2. Query PostgreSQL pgvector
    conn = db_pool.getconn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            # Cosine similarity formula: 1 - cosine_distance
            # In pgvector: embedding <=> %s::vector calculates cosine distance
            search_sql = """
                SELECT chunk_text, page_number, (1.0 - (embedding <=> %s::vector)) AS score
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            cur.execute(search_sql, (query_vector, query_vector, payload.top_k))
            rows = cur.fetchall()

        results = [
            ChunkResult(
                chunk_text=row[0],
                page_number=int(row[1]),
                score=round(float(row[2]), 4),
            )
            for row in rows
        ]
        return results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search failed: {str(e)}",
        )
    finally:
        db_pool.putconn(conn)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
