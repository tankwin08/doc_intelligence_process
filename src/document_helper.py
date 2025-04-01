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
import os
import json
from datetime import datetime

class DocumentHelper:
    """A helper class for processing various document types and extracting metadata.
    
    This class handles different file formats including PDF, DOCX, PPT, CSV, TXT, and images.
    It extracts content and metadata, then saves the processed documents in a structured format.
    """
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.csv': CSVLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.png': UnstructuredImageLoader,
        '.jpg': UnstructuredImageLoader,
        '.jpeg': UnstructuredImageLoader
    }
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the DocumentHelper with customizable chunking parameters.
        
        Args:
            chunk_size (int): The size of text chunks for processing
            chunk_overlap (int): The overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process_file(self, file_path: str) -> List[Document]:
        """Process a single file and return a list of documents with metadata.
        
        Args:
            file_path (str): Path to the file to process
            
        Returns:
            List[Document]: List of processed documents with metadata
            
        Raises:
            ValueError: If file type is not supported
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Load the document using appropriate loader
        loader_class = self.SUPPORTED_EXTENSIONS[file_ext]
        loader = loader_class(file_path)
        documents = loader.load()
        
        # Add additional metadata
        for doc in documents:
            doc.metadata.update({
                'source_file': os.path.basename(file_path),
                'file_type': file_ext,
                'processed_at': datetime.now().isoformat(),
                'total_pages': len(documents)
            })
        
        # Split documents into chunks
        return self.text_splitter.split_documents(documents)
    
    def process_files(self, file_paths: List[str]) -> List[Document]:
        """Process multiple files and return combined list of documents.
        
        Args:
            files (List[str]): List of file paths to process
            
        Returns:
            List[Document]: Combined list of processed documents
        """
        all_documents = []
        for file_path in file_paths:
            try:
                documents = self.process_file(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        return all_documents
    
    def save_to_json(self, documents: List[Document], output_path: str):
        """Save processed documents to a JSON file with metadata.
        
        Args:
            documents (List[Document]): List of processed documents
            output_path (str): Path to save the JSON file
        """
        docs_data = [{
            'content': doc.page_content,
            'metadata': doc.metadata
        } for doc in documents]
        
        with open(output_path, 'w') as f:
            json.dump(docs_data, f, indent=2)
    
    def load_from_json(self, input_path: str) -> List[Document]:
        """Load documents from a JSON file.
        
        Args:
            input_path (str): Path to the JSON file
            
        Returns:
            List[Document]: List of documents with metadata
        """
        with open(input_path, 'r') as f:
            docs_data = json.load(f)
        
        return [Document(
            page_content=doc['content'],
            metadata=doc['metadata']
        ) for doc in docs_data]


