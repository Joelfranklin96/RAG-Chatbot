# RAG Chatbot for Emerjence

A Retrieval-Augmented Generation (RAG) chatbot application that allows users to upload multiple PDF documents and interact with their content through natural language queries.

## Overview

This application combines the power of Large Language Models (LLMs) with vector search capabilities to provide an intelligent chatbot that can answer questions based on the content of uploaded PDF documents. The system uses a RAG architecture to enhance the quality and accuracy of responses by grounding them in the provided document context.

## Features

- **PDF Document Processing**: Upload and process multiple PDF files
- **Vector Embedding**: Convert document chunks into vector embeddings for semantic search
- **Context-Aware Responses**: Generate responses that take into account previous conversation history
- **Modern UI**: Clean, dark-themed interface with user profile integration
- **Real-time Chat**: Interactive chat interface for seamless conversation

## Technical Architecture

The application is built using the following technologies:

- **Frontend**: Streamlit for the web interface
- **Backend**: LangChain for orchestrating the RAG pipeline
- **Vector Database**: Pinecone for storing and searching vector embeddings
- **LLM**: OpenAI's GPT-4o for generating responses
- **Embedding Model**: OpenAI's `text-embedding-3-small` for creating vector embeddings

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root with your API keys:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

## Running the Application

```bash
streamlit run main.py
```

The application will be available at [http://localhost:8501](http://localhost:8501) by default.

## Usage

1. **Launch the application**
2. **Upload** one or more PDF documents using the file uploader in the sidebar
3. Click **"Submit & Process"** to analyze and index the documents
4. Once processing is complete, **enter your queries** in the text input field
5. View the chatbot's responses in the **chat interface**

## Implementation Details

### Document Processing Pipeline

1. PDF documents are uploaded and text is extracted.
2. The text is split into manageable chunks using `RecursiveCharacterTextSplitter`.
3. Each chunk is converted to a vector embedding using OpenAI's embedding model.
4. The embeddings are stored in a Pinecone vector database for efficient retrieval.

### Query Processing Pipeline

1. User queries are processed through a history-aware retriever.
2. The original query is rephrased using chat history to improve retrieval performance.
3. Relevant document chunks are retrieved from the vector database.
4. The LLM generates a response based on the retrieved context.

## Customization

- **Pinecone Index Name**: Modify the `INDEX_NAME` in `consts.py` to change the Pinecone index name.
- **Text Splitting**: Adjust chunk size and overlap in `get_text_chunks()` for different document types.
- **LLM Model**: Change the model by modifying the `model_name` parameter in `run_llm()`.
