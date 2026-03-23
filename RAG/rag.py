import os
from dotenv import load_dotenv
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

if __name__ == "__main__":
    load_dotenv()
    # Check/set API key from .env
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found! Add it to .env file in project root.")
        print("Example: GROQ_API_KEY=your_groq_key_here")
        exit(1)

def load_documents(docs_path="documents/", persist_dir="chroma_db/"):
    """
    Load all PDFs from docs_path (all 3 documents), split into chunks, create and persist Chroma vector store.
    Returns: Chroma vector store object.
    """
    docs_path = os.path.abspath(docs_path)
    persist_dir = os.path.abspath(persist_dir)
    documents = []
    pdf_files = []
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                pdf_files.append(file_path)
    
    print(f"Found {len(pdf_files)} PDF files in {docs_path}")
    for file_path in pdf_files:
        try:
            print(f"Loading {file_path}...")
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():  # skip empty pages
                    page_doc = Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1
                        }
                    )
                    documents.append(page_doc)
            doc.close()
            num_pages = len([d for d in documents if "source" in d.metadata and os.path.basename(d.metadata["source"]) == os.path.basename(file_path)])
            print(f"  Successfully loaded {num_pages} pages from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  FAILED to load {os.path.basename(file_path)}: {str(e)}")
    
    print(f"Total loaded {len(documents)} documents from {len(pdf_files)} PDFs")
    
    # Split into chunks
    chunks = split_documents(documents)
    
    # Create and persist vector store
    vector_store = create_vector_store(chunks, persist_dir)
    print(f"Vector store created and persisted at {persist_dir} with {vector_store._collection.count()} chunks from all documents")
    return vector_store

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # each chunk = ~1000 characters
        chunk_overlap=200,    # overlap keeps context between chunks
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content and len(c.page_content.strip()) > 10]
    print(f"Split into {len(chunks)} non-empty chunks")
    if not chunks:
        print("WARNING: No valid chunks. PDFs may be scanned images without text.")
    return chunks

def create_vector_store(chunks, persist_dir="chroma_db/"):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # free, runs locally
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Vector store created with {vector_store._collection.count()} chunks")
    return vector_store

def load_vector_store(persist_dir="chroma_db/"):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vector_store

def build_rag_chain(vector_store, groq_api_key):
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-8b-8192",  # free Llama 3 model via Groq
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an energy efficiency expert assistant. 
    Use the following context from building energy audit reports to answer the question.
    If the answer is not in the context, say "I don't have enough information in the reports to answer that."
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain

def query(rag_chain, vector_store, question):
    result = rag_chain.invoke(question)
    # To get sources, we need separate retriever call
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    sources = []
    if retriever:
        docs = retriever.invoke(question)
        sources = [doc.metadata.get("source", "unknown") for doc in docs]
    return {
        "answer": result.content,
        "sources": sources
    }
    
    print("1. Loading PDFs from documents/...")
    docs = load_documents()
    
    print("2. Splitting into chunks...")
    chunks = split_documents(docs)
    
    print("3. Loading vector store (builds if missing)...")
    try:
        vector_store = load_vector_store()
        print(f"Loaded existing vector store with {vector_store._collection.count()} vectors.")
    except Exception as e:
        print(f"Failed to load vector store ({e}), creating new...")
        vector_store = create_vector_store(chunks)
    
    print("4. Building RAG chain...")
    rag_chain = build_rag_chain(vector_store, groq_api_key)
    
    print("\nRAG ready! Ask questions about energy audit reports (type 'quit' to exit):")
    while True:
        question = input("\nQ: ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        result = query(rag_chain, vector_store, question)
        print(f"A: {result['answer']}")
        if result['sources']:
            print(f"Sources: {result['sources']}")

