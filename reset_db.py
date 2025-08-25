import os
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Get the Google API key from environment variables
google_api_key = os.getenv("GEMINI_API_KEY")
# Check if API key exists
if not google_api_key:
    print("GEMINI_API_KEY not found in .env file")
    exit(1)

# Import required modules for document processing and vector storage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def reset_vectorstore():
    """Reset the vector store by deleting existing data and creating new one."""
    # Define paths for ChromaDB persistence
    persist_directory = "./chroma_db"
    collection_name = "stock_market"
    
    # Delete existing vector store directory if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("Deleted existing vector store")
    
    # Create new vector store directory
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize embeddings model for converting text to vectors
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"  # Google's embedding model
    )
    
    # Load PDF document
    pdf_path = "/home/eagle/Downloads/Stock_Market_Performance_2024.pdf"
    
    # Verify PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print("Loading PDF...")
    # Load PDF using PyPDFLoader
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    print(f"PDF loaded successfully with {len(pages)} pages")
    
    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Maximum chunk size
        chunk_overlap=200     # Overlap between chunks for context continuity
    )
    splits = text_splitter.split_documents(pages)
    print(f"Split into {len(splits)} chunks")
    
    # Create new vector store with document chunks
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Vector store created successfully with {vectorstore._collection.count()} documents!")

# Run reset function when script is executed directly
if __name__ == "__main__":
    reset_vectorstore()