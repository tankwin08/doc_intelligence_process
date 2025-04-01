import os
import sys
from pathlib import Path

from typing import List, Dict, Any, Optional
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Import your existing document_helper functionality
# Adjust the import path as needed based on your project structure
from src.document_helper import DocumentHelper
# Instantiate DocumentHelper
doc_helper = DocumentHelper()

# Helper to load and process documents
def process_documents(files):
    global vector_store
    documents = []
    
    temp_dir = "temp_output"
    os.makedirs(temp_dir, exist_ok=True)
    
    for file_path in files:
        try:
            # Process file using DocumentHelper
            processed_docs = doc_helper.process_file(file_path)
            documents.extend(processed_docs)
            
            # Clean up
            #os.remove(file_path)
            
        except Exception as e:
            app.logger.error(f"Error processing file {file.filename}: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)
    
    if not documents:
        raise ValueError("No documents were successfully processed")
    
    # Create or update vector store
    if vector_store is None:
        vector_store = FAISS.from_documents(documents, embeddings)
    else:
        vector_store.add_documents(documents)
        
    # Save processed documents to JSON for reference
    output_path = os.path.join(temp_dir, "processed_documents.json")
    doc_helper.save_to_json(documents, output_path)

    return output_path, documents, vector_store
