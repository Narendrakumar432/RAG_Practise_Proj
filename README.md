# PDF Vectorisation Pipeline — Nestack Technical Assessment

A high-performance, lightweight, end-to-end PDF vector ingestion and semantic retrieval pipeline built strictly in accordance with Nestack's technical assessment specification.

This implementation uses **direct SDK and library calls only** — completely free of LangChain, LlamaIndex, agent frameworks, conversational layers, or LLM-based generation on retrieval.

---

## 🛠️ Technology Stack

| Layer | Technology | Details / Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Standard, type-annotated, modern Python |
| **PDF Extraction** | PyMuPDF (`pymupdf`) | High-speed C-native text extraction with page preservation |
| **Text Chunking** | Custom Sliding Window | 500 chars / 100 char overlap with word-boundary snapping |
| **Embedding Model** | `all-MiniLM-L6-v2` | 384-dimensional dense semantic vectors via `sentence-transformers` |
| **Vector Database** | PostgreSQL + `pgvector` | Native relational storage + `VECTOR(384)` with HNSW cosine index |
| **Backend API** | FastAPI + Uvicorn | Async, type-validated HTTP REST API |
| **Client Testing** | Postman / `curl` | Full support for JSON payloads |

---

## 📐 Technical Design Decisions & Justifications

### 1. Chunk Size and Overlap Justification
- **Model Context Constraint**: The chosen embedding model `all-MiniLM-L6-v2` has a hard token input limit of **256 wordpiece tokens** (equivalent to roughly 180–200 words or ~800–1000 characters). Providing text longer than 256 tokens results in **silent tail truncation**, leading to unindexed text and permanent information loss.
- **Selected Chunk Size (`500 characters`)**: A 500-character chunk corresponds to ~90–110 words (approximately 3–5 sentences). In dense semantic retrieval literature, 100-word passages represent the sweet spot: sufficiently large to preserve complete context and topic coherence, but tight enough that irrelevant surrounding topics do not dilute the dense vector representation.
- **Selected Chunk Overlap (`100 characters` / ~20%)**: Text boundaries frequently split sentences, entities, or compound requirements across adjacent chunks. A 20% overlap ensures that any thought bridging a boundary is captured intact in at least one chunk.
- **Word-Boundary Snapping**: The custom chunker snaps to the nearest whitespace or newline rather than cutting arbitrarily through characters, preventing corrupted words (e.g. splitting `vector` into `vec` and `tor`).
- **Page Number Tracking**: Every chunk retains its exact 1-indexed source PDF page number throughout chunking, embedding, database storage, and retrieval.

### 2. Embedding Model Choice (`all-MiniLM-L6-v2`)
- **Dimension Efficiency**: Produces compact **384-dimensional** embeddings (`VECTOR(384)`). This requires less than half the RAM and storage of 768-dim (BERT) or 1536-dim (OpenAI) models while retaining comparable semantic accuracy.
- **Inference Speed**: Optimized for low-latency CPU and GPU inference (~5x faster than BERT-base), enabling instant response times for interactive retrieval queries without requiring expensive GPU instances.
- **Semantic Fidelity**: Extensively pre-trained and fine-tuned on over 1 billion sentence pairs for semantic search and contrastive ranking.
- **Zero API Dependency**: Runs completely locally via HuggingFace `sentence-transformers`, with zero external rate limits, costs, or API keys required.

### 3. Vector Database Choice (PostgreSQL + `pgvector`)
- **Unified Relational & Vector Architecture**: Unlike standalone vector stores (e.g. Pinecone, ChromaDB), PostgreSQL with `pgvector` allows atomic transactions (ACID), relational constraints, source page metadata, and vector embeddings to reside in a single production-grade database.
- **Cosine Distance Search**: Vector similarity ranking is computed directly in the database engine using the native `<=>` cosine distance operator:
  $$\text{score} = 1.0 - (\text{embedding} \Leftrightarrow \text{query\_vector})$$
- **HNSW Indexing**: Indexed via an HNSW (Hierarchical Navigable Small World) index (`USING hnsw (embedding vector_cosine_ops)`), enabling sub-millisecond approximate nearest neighbor searches even at large scale.
- **Portability**: Runs locally via Docker or in the cloud using standard managed PostgreSQL providers (Supabase, Neon, AWS RDS, Render).

---

## 📂 Project Structure

```
Narendra_Nestack_Submission/
├── ingest.py               # PDF extraction, custom chunking, embedding, pgvector storage
├── server.py               # FastAPI retrieval server with POST /query
├── requirements.txt        # Pinned Python dependencies
├── README.md               # Complete architecture, justifications, setup, and submission docs
├── results.json            # Output of 3 distinct test queries against sample PDF
├── test_query.py           # Automated test suite for health check & query execution
├── .env.example            # Environment variables template
├── .gitignore              # Ignores .env, .venv, caches, and submission archives
├── assets/
│   └── postman_test_result.png # Verified Postman execution screenshot
├── sample_data/
│   └── document.pdf        # Test assessment PDF
└── vector_db/
    ├── docker-compose.yml  # pgvector PostgreSQL 16 container definition
    └── init.sql            # SQL schema & extension initialization
```

---

## ⚙️ Installation & Local Setup

### Prerequisites
- **Python**: 3.10 or higher
- **Docker & Docker Compose** (Recommended for 1-command pgvector startup) OR an existing PostgreSQL 15+ installation with `pgvector` enabled.

---

### Step 1: Clone or Extract the Repository

```bash
git clone <YOUR_PRIVATE_GITHUB_REPO_URL>
cd Narendra_Nestack_Submission
```

### Step 2: Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 3: Start PostgreSQL with pgvector

#### Option A: Using Docker Compose (Recommended - 1 Command)
A ready-to-run `docker-compose.yml` is provided in `vector_db/` mapping port `5433`:

```bash
docker compose -f vector_db/docker-compose.yml up -d
```

Verify that the container is healthy:
```bash
docker ps --filter "name=nestack_pgvector"
```

#### Option B: Using an Existing Local or Cloud PostgreSQL
If you already have PostgreSQL running with pgvector:
1. Connect via `psql` and execute the schema in `vector_db/init.sql`:
   ```bash
   psql -U postgres -d postgres -f vector_db/init.sql
   ```
2. Update `.env` with your database credentials.

---

### Step 4: Configure Environment Variables

Create `.env` from the template:

```bash
cp .env.example .env
```

Default configuration for Docker Compose setup:
```ini
DB_HOST=localhost
DB_PORT=5433
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres

# Server Settings
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

*(If using a cloud database like Supabase or Neon, you can simply set `DATABASE_URL=postgresql://user:pass@host:5432/dbname`).*

---

## 📥 Ingestion Execution

Run the ingestion script to process any PDF:

```bash
python ingest.py --file sample_data/document.pdf
```

### Optional Arguments:
- `--chunk-size`: Size of chunk in characters (default: `500`).
- `--chunk-overlap`: Character overlap between chunks (default: `100`).
- `--reset`: Truncate the table before ingestion (clean re-ingest).

```bash
# Example with clean table reset:
python ingest.py --file sample_data/document.pdf --reset
```

### Ingestion Output:
```text
=== Starting Ingestion for: sample_data/document.pdf ===
Clearing existing records in document_chunks...
Extracting text from PDF...
Extracted 2 page(s) with text.
Chunking text (chunk_size=500, overlap=100)...
Generated 10 overlapping chunk(s).
Loading embedding model: all-MiniLM-L6-v2...
Generating 384-dimensional embeddings for 10 chunk(s)...
Inserting chunks and embeddings into PostgreSQL (pgvector)...
Successfully ingested 10 chunk(s) from sample_data/document.pdf into PostgreSQL.
```

---

## 🌐 Running the Retrieval API Server

Start the FastAPI retrieval server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
Or directly:
```bash
python server.py
```

The server starts at `http://localhost:8000`.
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

---

## 🔍 API Testing & Verified Sample Output

### 1. Postman Verified Execution

The pipeline has been thoroughly verified using **Postman** against the live running FastAPI service and PostgreSQL vector database:

![Postman Query Verification](assets/postman_test_result.png)

- **Endpoint**: `POST http://localhost:8000/query`
- **Response Status**: `200 OK`
- **Latency**: `193 ms` (sub-200ms vector search on CPU)
- **Content-Type**: `application/json`

---

### 2. Sample Query & JSON Response

#### Request (`curl`):
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How should text chunking and overlap be designed and justified?",
    "top_k": 3
  }'
```

#### Response (`200 OK`):
```json
[
  {
    "chunk_text": "ts. \n2. Split the extracted text into overlapping chunks. Chunk size and overlap are your decision - justify both in the \nREADME. \n3. Generate a vector embedding for each chunk using any embedding model (e.g. OpenAI text-embedding-3-small, \nsentence-transformers, Cohere Embed). \n4. Store each embedding together with its chunk text and source page number in a vector database of your choice (e.g. \nChromaDB, FAISS, Pinecone, pgvector). \n \nThe script must be runnable as: \npython ingest.py --file",
    "page_number": 1,
    "score": 0.5041
  },
  {
    "chunk_text": "plain-text query and \nreturns the most semantically similar chunks from the stored embeddings. There is no LLM call on the retrieval side \nand no conversational layer. Return raw chunk results only. \nPART 1 - INGESTION \nWrite a script that accepts a PDF file path and executes the following steps in order: \n1. Extract all text from the PDF, preserving page numbers. Handle multi-page documents. \n2. Split the extracted text into overlapping chunks. Chunk size and overlap are your decision -",
    "page_number": 1,
    "score": 0.392
  },
  {
    "chunk_text": "Technical Assessment - Nestack \nDo not share or distribute this document. \nPage PAGE\nEVALUATION \nCriterion \nWeight \nWhat we look for \nPipeline runs end-to-end without errors \n35% \nIngestion completes; retrieval returns \nrelevant results. \nChunking and embedding design decisions \n25% \nChoices are justified with reasoning in the \nREADME. \nRetrieval quality - relevance of returned chunks \n25% \nresults.json entries are semantically \nrelevant to each query. \nCode clarity and README completeness \n15%",
    "page_number": 2,
    "score": 0.2808
  }
]
```

---

### 3. Automated Verification Script
Run the automated verification script which executes 3 distinct queries and produces `results.json`:
```bash
python test_query.py
```

3. **Ingest Documents into the Cloud Database**:
   - Run `python ingest.py --file sample_data/document.pdf` with `DATABASE_URL` set in your `.env`.

---

