import streamlit as st

def render_about():
    """Render the about page"""
    st.header("About Document Intelligence System")
    st.write("""
    This application allows you to upload documents and ask questions about their content using AI.
    
    ### Features
    - Upload multiple document types (PDF, CSV, TXT, etc.)
    - Process documents using LangChain and Ollama
    - Interactive chat interface to ask questions about document content
    - Batch processing for multiple queries
    - Support for different LLM models via Ollama
    
    ### Supported File Types
    - PDF (.pdf)
    - CSV (.csv)
    - Text (.txt)
    - Word Documents (.docx, .doc)
    - PowerPoint (.ppt, .pptx)
    - Images (.jpg, .jpeg, .png)
    
    ### How to Use
    1. Upload one or more documents
    2. Process the documents
    3. Ask questions about the content
    
    ### Technology
    This application uses RAG (Retrieval Augmented Generation) with LangChain and Ollama to provide accurate answers based on your documents.
    """)
    
    st.write("### Created by")
    st.write("tankwin08, if you are interested in building a more professional document intelligence system for your organization, please contact me at: tankchow12@gmail.com")
    
    st.write("### License")
    st.write("This project is licensed under the MIT License")