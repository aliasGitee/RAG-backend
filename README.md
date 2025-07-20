# RAG Backend

This project provides a simple RAG (Retrieval-Augmented Generation) backend service. It leverages the `deepdoc` library for sophisticated layout analysis and utilizes Vision Language Models (VLMs) for deep content understanding.

## Core Functionality

The primary goal of this backend is to intelligently parse complex documents, such as academic papers, by recognizing their structure and content. It identifies and processes distinct elements to build a high-quality knowledge base for retrieval.

### Key Features

- **Layout-Aware Parsing**: Powered by `deepdoc`, the service accurately identifies the layout of documents, distinguishing between different content blocks.
- **Element Recognition**: It can specifically recognize and isolate:
  - **Text Paragraphs**: Standard text blocks.
  - **Tables & Captions**: Detects tabular data and their associated titles/captions.
  - **Formulas & Equations**: Identifies mathematical and scientific formulas, along with their numbering.
- **VLM-Powered Analysis**: Utilizes a Vision Language Model (VLM) for:
  - **Text Parsing**: Extracts and cleans text from identified blocks.
  - **Chart & Table Understanding**: Analyzes images of tables and figures to understand their structure and content, converting visual data into textual descriptions.
- **API-Driven**: Exposes simple FastAPI endpoints for processing documents, generating embeddings, and performing retrieval.

## Getting Started

### Dependencies

Install the necessary Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- **`POST /generate-chunks`**: 
  - Processes an input document.
  - Performs layout analysis and uses a VLM to extract text, tables, and figures.
  - Saves the processed chunks to the specified `root_path`.

- **`POST /generate-embeddings`**: 
  - Takes the processed chunks from the previous step.
  - Generates vector embeddings for each chunk using the specified model.
  - Persists the embeddings to a vector store (ChromaDB).

- **`POST /generate-retriever`**: 
  - Accepts a user query (`prompt`).
  - Retrieves relevant chunks from the vector store.
  - Uses a VLM to synthesize an answer based on the retrieved context and images.