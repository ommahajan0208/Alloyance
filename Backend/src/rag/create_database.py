from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # <-- NEW

import os
import shutil
import glob

# Using local embeddings instead of OpenAI paid api
CHROMA_PATH = "chroma"
DATA_PATH = os.path.join("..", "..", "data", "rag_data")

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    documents = []
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))  # Load all PDFs
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDF files.")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first, but safely
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except PermissionError:
            print("Chroma DB is locked or used by other kernel")
            return

    # Use free, local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a new DB from the documents
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    db = None  # Explicitly drop reference to release file handles
    print(f" Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
