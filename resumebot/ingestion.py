from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_pdf(pdf_path):
    # Load PDF and extract text
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text recursively into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Adjusted chunk size and overlap
    split_documents = text_splitter.split_documents(documents)

    # Create a vector store for the split text
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(split_documents, embeddings)

    return vectorstore

if __name__ == '__main__':
    pdf_path = "path_to_your_pdf.pdf"
    vectorstore = ingest_pdf(pdf_path)
    print("Vector Store created for the PDF.")