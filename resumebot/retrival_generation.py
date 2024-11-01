from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from resumebot.ingestion import ingest_pdf

load_dotenv()

# Load Hugging Face model with an increased max_length for longer responses
def load_hf_model():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_token = os.getenv("HF_TOKEN")
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=400, temperature=0.7, token=hf_token)  # Increased max_length
    return llm

# Create the RetrievalQA chain with updated retrieval settings
def create_retrieval_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})  # Retrieve multiple similar chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def generate_answer(vectorstore, query):
    llm = load_hf_model()
    qa_chain = create_retrieval_qa_chain(llm, vectorstore)
    response = qa_chain({"query": query})
    return response["result"], response["source_documents"]

if __name__ == '__main__':
    # Example usage
    pdf_path = "path_to_your_pdf.pdf"
    vectorstore = ingest_pdf(pdf_path)
    result, source_docs = generate_answer(vectorstore, "What is the content of the PDF?")
    print(f"Answer: {result}")
    print(f"Source Documents: {source_docs}")