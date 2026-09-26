# AI Research Assistant

A local, PDF-based research assistant built with Streamlit, sentence-transformer embeddings, ChromaDB, Ollama, and LangGraph. Upload research documents, retrieve relevant text chunks, and ask questions grounded in the retrieved context.

## Project overview

The application extracts text from uploaded PDFs, splits it into overlapping chunks, embeds those chunks with `all-MiniLM-L6-v2`, and stores them in the persistent ChromaDB collection `research_documents`. For each chat question, it retrieves the three nearest chunks and asks the local Ollama `llama3.2` model to answer using that context.

The Streamlit application calls the RAG generator directly. A LangGraph tool-calling agent is also included as a separate command-line entry point; it is not currently the agent used by the Streamlit chat UI.

## Problem statement

Research material is often lengthy and difficult to search manually. This project provides a simple local workflow for indexing text-based PDFs and asking questions against their contents, with retrieved document text supplied to the language model as answer context.

## Key features

- Upload and process multiple PDFs in one Streamlit session.
- Extract PDF text and split it into 500-character chunks with 50-character overlap.
- Generate embeddings with the existing `all-MiniLM-L6-v2` model.
- Store chunks in the persistent `research_documents` ChromaDB collection.
- Attach each source filename as chunk metadata.
- Use deterministic IDs and ChromaDB upserts so reprocessing the same filename and file contents updates its chunks without removing other documents.
- Retrieve the three nearest chunks for a question.
- Generate answers with the local Ollama `llama3.2` model and pass Streamlit conversation history as context.
- Provide a separate LangGraph agent with a `research_search` tool.
- Run a basic five-question retrieval evaluation based on expected topic phrases.
- Display user-facing errors for common PDF, embedding, retrieval, and Ollama failures.

## Architecture

### Streamlit application and RAG flow

The implemented Streamlit path uses the RAG generator directly:

```text
PDF indexing:
User → Streamlit uploader → PDF loader → Chunking → Embeddings → ChromaDB

Question answering:
User question + session conversation history
  → Streamlit
  → Retriever (embedding + ChromaDB query)
  → Top 3 document chunks
  → RAG generator
  → Ollama Llama 3.2
  → Answer in Streamlit chat
```

### Separate LangGraph agent architecture

The LangGraph agent is available independently through `src.agent.agent`:

```text
User message
  → LangGraph agent node (ChatOllama: llama3.2)
  → model tool call, when selected
  → research_search tool
  → RAG Search Tool (src.agent.rag_tool)
  → Retriever → Embedding model → ChromaDB
  → Retrieved text returned to the agent
  → LangGraph agent node
  → Final answer
```

The model decides whether to call `research_search`; tool use is not guaranteed for every prompt. The Streamlit UI currently calls `generate_answer` directly rather than invoking this graph.

### Conversation-memory flow

```text
Streamlit session_state["messages"]
  → render prior user/assistant messages
  → append the new user question
  → pass message history to generate_answer
  → include history with retrieved context in the Ollama request
  → append the assistant answer to session_state["messages"]
```

Conversation history lives in the current Streamlit session; it is not persisted to ChromaDB or a separate database.

## Complete RAG pipeline

1. A user uploads one or more PDF files in Streamlit.
2. Streamlit writes each file under `data/documents/` and calls `ingest_pdf`.
3. `load_pdf` extracts page text with `pypdf`.
4. `split_text` creates overlapping chunks using `RecursiveCharacterTextSplitter`.
5. `create_embeddings` converts chunks to vectors using `all-MiniLM-L6-v2`.
6. Ingestion computes IDs from the filename and PDF bytes, adds `filename` metadata, and upserts each chunk into ChromaDB.
7. For a question, `search` embeds the query with the same embedding model.
8. ChromaDB returns the three nearest chunks from the shared collection.
9. `generate_answer` combines those chunks with the question and conversation history and sends the messages to Ollama `llama3.2`.
10. The answer is rendered and added to the current Streamlit session history.

## AI Agent architecture

`src/agent/agent.py` builds a LangGraph `StateGraph` with an agent node and a tool node. The agent uses LangChain's `ChatOllama` configured for `llama3.2`. `tools_condition` routes to the tool node when the model returns a tool call. The graph loops from the tool node back to the agent so the model can use the retrieved text in its final response.

This graph is a separate entry point; the Streamlit chat currently uses `src.rag.generator.generate_answer` instead.

## RAG Search Tool

`src/agent/rag_tool.py` exposes `search_documents_tool(question)`. It calls the shared `src.retrieval.search.search` function with `n_results=3`, returns the retrieved text joined together, or returns a no-relevant-information message when no documents are found. `research_search` in `src/agent/agent.py` wraps this function as a LangChain tool.

## Technologies used

| Technology | Use |
| --- | --- |
| Python | Application and pipeline implementation |
| Streamlit | PDF upload and chat UI |
| pypdf | PDF text extraction |
| LangChain Text Splitters | Recursive text chunking |
| LangChain Core | Message and tool abstractions |
| Sentence Transformers | `all-MiniLM-L6-v2` text embeddings |
| ChromaDB | Persistent vector storage and similarity search |
| Ollama | Local `llama3.2` inference |
| LangChain Ollama integration | Chat model and tool binding in the agent |
| LangGraph | Tool-calling agent workflow |

## Project folder structure

```text
ai-research-assistant/
├── app/
│   └── app.py                    # Streamlit application
├── data/
│   ├── documents/                # Uploaded and sample PDFs
│   └── chroma/                   # Persistent ChromaDB data (created at runtime)
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── agent.py              # LangGraph tool-calling agent
│   │   └── rag_tool.py           # Retrieval tool used by the agent
│   ├── embeddings/
│   │   └── embedder.py            # Sentence Transformer embeddings
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_rag.py        # Five-question retrieval evaluation
│   ├── ingestion/
│   │   ├── chunker.py             # Text splitting
│   │   ├── ingest.py              # PDF ingestion orchestration
│   │   └── pdf_loader.py          # PDF text extraction
│   ├── rag/
│   │   ├── __init__.py
│   │   └── generator.py           # Retrieval-augmented answer generation
│   └── retrieval/
│       ├── search.py              # Query embedding and retrieval
│       └── vector_store.py        # ChromaDB collection operations
├── requirements.txt
└── README.md
```

## Installation

These commands are for Windows PowerShell and assume the repository is at `C:\Users\DINESH\ai-research-assistant`.

```powershell
cd C:\Users\DINESH\ai-research-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the venv's executables directly, for example `./venv/Scripts/python.exe` and `./venv/Scripts/streamlit.exe`.

### Install Python dependencies

The dependencies used by this project are listed in `requirements.txt`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The sentence-transformer model may be downloaded the first time embeddings are generated. No package installation is needed after the environment has been set up.

### Install and run Ollama

Install Ollama for Windows using the [official Ollama download page](https://ollama.com/download/windows), then open PowerShell and download the model used by the application:

```powershell
ollama pull llama3.2
ollama run llama3.2
```

Ollama's Windows app normally runs its local service in the background. If you run Ollama as a service manually, use `ollama serve` in a separate terminal. The application expects the local Ollama service and the `llama3.2` model to be available.

## Run the Streamlit application

From the project root, with the virtual environment active:

```powershell
streamlit run app\app.py
```

Streamlit prints a local URL in the terminal. Open it in a browser to use the app.

## Upload PDFs

1. Open the Streamlit application.
2. Select one or more PDF files in **Upload a PDF**.
3. Choose **Process PDF**.
4. Review the filename, extracted character count, chunk count, and processing status shown for each file.

PDFs are copied into `data/documents/`. Their chunks are added to the shared ChromaDB collection and remain available when another PDF is processed. Reprocessing a PDF with the same filename and bytes uses the same chunk IDs, so it updates those records instead of creating duplicate IDs. Scanned pages without extractable text are not OCR-processed.

## Ask questions

Enter a question in the chat input, for example:

- What are embeddings?
- What is retrieval augmented generation?
- How does semantic search work?
- What is text chunking?
- What are the benefits of vector databases?

The application retrieves up to three chunks from the shared collection and asks the model to answer using the document context. If there are no matching chunks, the RAG generator returns a message asking the user to upload and process a PDF.

## How the RAG pipeline works

Retrieval-augmented generation (RAG) adds document search to a language-model request. This project first searches the user's indexed PDF chunks, then places those retrieved chunks in the prompt sent to the language model. The model is instructed to use the provided document context and to say when the answer is not available there.

### Embeddings

An embedding is a numeric vector representing text. The same `all-MiniLM-L6-v2` model embeds both document chunks and user questions. ChromaDB compares these vectors to retrieve chunks that are semantically similar to the question. Embeddings indicate similarity; they do not guarantee that a retrieved chunk contains a complete or correct answer.

### ChromaDB

The application uses `chromadb.PersistentClient` at `./data/chroma` and the existing `research_documents` collection. Each chunk is stored with its text, embedding, deterministic ID, and source filename metadata. Ingestion uses upsert, so matching IDs are replaced while unrelated records remain in the collection. Retrieval queries the collection without filtering on filename, allowing chunks from all ingested PDFs to be considered together.

### LangGraph

LangGraph is used in the separate agent module to define the agent/tool loop. The graph starts at the agent node, routes model-selected tool calls to the tool node, then returns tool results to the agent. It is not currently connected to the Streamlit chat flow.

## RAG evaluation

Run the lightweight retrieval evaluation with:

```powershell
python -m src.evaluation.evaluate_rag
```

The script searches for five fixed questions with `n_results=3`. A question passes when its expected phrase (`embeddings`, `retrieval`, `chunking`, `semantic search`, or `vector databases`) appears case-insensitively in the returned chunk text. This is a basic phrase-presence check, not a measurement of answer quality, ranking relevance, or factual correctness. Its results depend on which PDFs have been ingested.

## Command-line entry points

From the project root:

```powershell
python -m src.rag.generator
python -m src.agent.rag_tool
python -m src.agent.agent
python -m src.evaluation.evaluate_rag
```

`src.rag.generator` defines the `generate_answer` function but has no module-level demo, so running it as a module only imports the code. The RAG Search Tool and LangGraph agent modules include simple `__main__` examples and make retrieval/model calls.

On Windows terminals using a legacy encoding, the LangGraph agent's tool-call display may fail when printing its wrench emoji. Set UTF-8 output before running it in that environment:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m src.agent.agent
```

## Limitations

- PDF extraction is text-based; there is no OCR for scanned pages.
- Chunk size, overlap, and retrieval count are fixed in code.
- Retrieval quality depends on the embedding model and indexed document text; there is no reranker or relevance threshold.
- The evaluation checks literal topic phrases in retrieved chunks and does not judge answer quality.
- Streamlit conversation history lasts only for the active session.
- The Streamlit UI calls the direct RAG generator, while the LangGraph agent is a separate workflow.
- Ollama and the `llama3.2` model must be installed and available locally for answer generation and agent execution.
- The embedding model may require network access on its first use if it is not cached.

## Future improvements

- Add OCR support for scanned PDFs.
- Make chunking and retrieval parameters configurable.
- Add stronger retrieval evaluation with relevance judgments and answer-quality measures.
- Persist or export conversation sessions when needed.
- Allow users to inspect the source filename and text for each retrieved passage.
- Add tests around ingestion, duplicate processing, retrieval, and agent tool routing.
