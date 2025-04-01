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

# Import DocumentHelper from the same directory
from src.document_helper import DocumentHelper

# Define the DocumentHelper class here to make it available for import
class DocumentHelper:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        # Add tracking for thinking process
        self.thinking_process = ""
        
    def capture_thinking(self, thought):
        """Capture thinking process for later display"""
        self.thinking_process += thought + "\n"
        
    def get_thinking_process(self):
        """Get the current thinking process"""
        return self.thinking_process
        
    def clear_thinking_process(self):
        """Clear the thinking process"""
        self.thinking_process = ""
    
    def process_file(self, file_path):
        """Process a file and return a list of Document objects"""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.ppt', '.pptx']:
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_extension in ['.jpg', '.jpeg', '.png']:
                loader = UnstructuredImageLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            split_docs = self.text_splitter.split_documents(documents)
            return split_docs
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def save_to_json(self, documents, output_path):
        """Save processed documents to JSON for reference"""
        import json
        
        docs_data = []
        for doc in documents:
            docs_data.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        with open(output_path, 'w') as f:
            json.dump(docs_data, f, indent=2)

# Initialize DocumentHelper
doc_helper = DocumentHelper()

# Global variable for vector store
vector_store = None

# Helper to load and process documents
def process_documents(files, embeddings=None, doc_helper=None):
    global vector_store
    documents = []
    
    # Use provided embeddings or create default
    if embeddings is None:
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    
    # Use provided doc_helper or use the global one
    if doc_helper is None:
        doc_helper = globals()['doc_helper']
    
    temp_dir = "temp_output"
    os.makedirs(temp_dir, exist_ok=True)
    
    for file_path in files:
        try:
            # Process file using DocumentHelper
            processed_docs = doc_helper.process_file(file_path)
            documents.extend(processed_docs)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
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

    return vector_store
