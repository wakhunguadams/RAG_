import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Get the Google API key from environment variables
google_api_key = os.getenv("GEMINI_API_KEY")
# Check if API key exists
if not google_api_key:
    print("GEMINI_API_KEY not found in .env file")
    exit(1)
# Set the API key in environment for LangChain to use
os.environ["GOOGLE_API_KEY"] = google_api_key

# Import required LangChain modules
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def initialize_rag_system():
    """Initialize the RAG system by loading documents and creating vector store."""
    print("Initializing RAG system...")
    
    # Initialize embeddings model for converting text to vectors
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    
    # Define paths and collection name for ChromaDB
    persist_directory = "./chroma_db"
    collection_name = "stock_market"
    
    # Try to load existing vector store to avoid reprocessing PDF
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # Check if vector store has documents already
        if vectorstore._collection.count() > 0:
            print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
            return vectorstore, embeddings
        else:
            print("Existing vector store is empty, will re-create it")
    except Exception as e:
        print(f"Could not load existing vector store: {e}")
        print("Will create a new one")
    
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
    
    return vectorstore, embeddings

def create_rag_chain(vectorstore, llm):
    """Create the RAG chain for question answering."""
    # Create retriever from vector store with similarity search
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    # Create prompt template for the LLM
    template = """
    You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024.
    Use the following context to answer the question at the end. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    # Create prompt from template
    prompt = PromptTemplate.from_template(template)
    
    # Function to format retrieved documents into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the RAG chain:
    # 1. Take question and retrieve relevant documents
    # 2. Format the documents
    # 3. Combine with question in prompt
    # 4. Send to LLM
    # 5. Parse output as string
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def ask_question(rag_chain, question):
    """Ask a question and get an answer from the RAG system."""
    try:
        # Invoke the RAG chain with the question
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        # Handle any errors during question processing
        return f"Error occurred: {str(e)}"

def main():
    """Main function to run the RAG agent."""
    print("=== Stock Market RAG Agent ===")
    
    try:
        # Initialize the RAG system (load documents, create vector store)
        vectorstore, embeddings = initialize_rag_system()
        
        # Initialize the LLM (Google's Gemini model)
        print("Initializing LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Fast and efficient model
            temperature=0                # Deterministic responses
        )
        
        # Create the RAG chain for question answering
        print("Creating RAG chain...")
        rag_chain = create_rag_chain(vectorstore, llm)
        print("RAG system ready!")
        
        # Print usage instructions
        print("\nAsk questions about the 2024 stock market performance.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        # Check if question was provided as command line argument
        if len(sys.argv) > 1:
            # If question provided as command line argument
            question = " ".join(sys.argv[1:])
            answer = ask_question(rag_chain, question)
            print("=== ANSWER ===")
            print(answer)
        else:
            # Interactive mode - continuously ask for questions
            while True:
                try:
                    # Get question from user
                    question = input("What is your question: ")
                    # Check for exit commands
                    if question.lower() in ['exit', 'quit']:
                        print("Goodbye!")
                        break
                    
                    # Skip empty questions
                    if question.strip() == "":
                        continue
                        
                    # Get answer from RAG system
                    answer = ask_question(rag_chain, question)
                    print("\n=== ANSWER ===")
                    print(answer)
                    print()  # Extra line for readability
                except KeyboardInterrupt:
                    # Handle Ctrl+C gracefully
                    print("\nGoodbye!")
                    break
                except EOFError:
                    # Handle EOF (Ctrl+D) gracefully
                    print("\nGoodbye!")
                    break
                    
    except Exception as e:
        # Handle any initialization errors
        print(f"Error initializing RAG system: {e}")
        sys.exit(1)

# Run main function when script is executed directly
if __name__ == "__main__":
    main()