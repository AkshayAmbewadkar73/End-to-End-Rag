from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
from resumebot.retrival_generation import generate_answer
from resumebot.ingestion import ingest_pdf

app = Flask(__name__)

load_dotenv()

# Global variable to store the vector store once PDF is processed
vectorstore = None

# Load the PDF and create vector store
@app.route("/upload", methods=["POST"])
def upload_pdf():
    global vectorstore
    if 'file' not in request.files:
        return "No file uploaded!", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file!", 400
    
    # Save the uploaded PDF
    pdf_path = file.filename
    file.save(pdf_path)
    
    # Ingest the PDF and create a vector store
    vectorstore = ingest_pdf(pdf_path)
    return "PDF uploaded and processed successfully!"

# Generate response based on the query
@app.route("/get", methods=["POST"])
def get_answer():
    if vectorstore is None:
        return "No PDF has been processed yet!", 400
    
    msg = request.form["msg"]
    
    if not msg:
        return "No query provided!", 400
    
    result, source_docs = generate_answer(vectorstore, msg)
    
    # Return the answer as JSON
    return jsonify({
        "answer": result,
        "source_docs": [doc.page_content for doc in source_docs]
    })

@app.route("/")
def index():
    return render_template('chat1.html')  # Assuming you have a chat.html file

if __name__ == '__main__':
    app.run(debug=True)