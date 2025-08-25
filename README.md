# Stock Market RAG Agent

This is a Retrieval-Augmented Generation (RAG) agent that can answer questions about the Stock Market Performance in 2024 based on a provided PDF document.

## Features

- Uses Google's Gemini model for natural language understanding
- Employs ChromaDB for efficient document retrieval
- Provides accurate answers based on the content of the PDF
- Handles questions outside the PDF scope gracefully

## Requirements

- Python 3.8+
- Google Gemini API key
- PDF document about stock market performance

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up your Google Gemini API key in the `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Interactive Mode
Run the agent in interactive mode:
```bash
python agent.py
```

Then type your questions when prompted.

### Command Line Mode
Ask a single question directly from the command line:
```bash
python agent.py "What was the S&P 500 return in 2024?"
```

## Code Structure

### agent.py
Main application file containing:
- Environment setup and API key loading
- PDF document loading and processing
- Vector store initialization and management
- RAG chain creation for question answering
- Interactive and command-line interfaces

### reset_db.py
Utility script to reset the vector database:
- Deletes existing ChromaDB storage
- Reprocesses the PDF document
- Creates fresh vector embeddings

### Key Components

1. **Document Loading**: Uses PyPDFLoader to extract text from PDF
2. **Text Splitting**: RecursiveCharacterTextSplitter breaks text into manageable chunks
3. **Embeddings**: GoogleGenerativeAIEmbeddings converts text to vectors
4. **Vector Store**: ChromaDB stores and retrieves document embeddings
5. **LLM**: ChatGoogleGenerativeAI (Gemini) generates natural language responses
6. **RAG Chain**: Combines retrieval and generation for accurate answers

## How It Works

1. The PDF document is loaded and split into chunks
2. These chunks are embedded using Google's embedding model and stored in ChromaDB
3. When a question is asked, relevant chunks are retrieved from the database
4. The question and relevant context are sent to the Gemini model for answer generation

## Example Questions

- "What was the S&P 500 return in 2024?"
- "How did the Nasdaq Composite perform in 2024?"
- "Which index had the best performance in 2024?"