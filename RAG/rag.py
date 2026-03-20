import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def load_documents(docs_path="documents/"):
    documents = []
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                loader = PyMuPDFLoader(file_path)
                documents.extend(loader.load())
    print(f"Loaded {len(documents)} pages from PDFs")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # each chunk = ~1000 characters
        chunk_overlap=200,    # overlap keeps context between chunks
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
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

if __name__ == "__main__":
    # Check/set API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: Set GROQ_API_KEY env var first!")
        print("Windows: $env:GROQ_API_KEY='your_key' then rerun")
        print("Or export GROQ_API_KEY=your_key (Linux/Mac)")
        exit(1)
    
    print("1. Loading PDFs from documents/...")
    docs = load_documents()
    
    print("2. Splitting into chunks...")
    chunks = split_documents(docs)
    
    print("3. Loading vector store (builds if missing)...")
    try:
        vector_store = load_vector_store()
        print("Loaded existing vector store.")
    except:
        print("Creating new vector store...")
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
