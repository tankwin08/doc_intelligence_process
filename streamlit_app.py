import streamlit as st
import json
import os
from io import BytesIO
import tempfile
import sys
import time

# Add the parent directory to the path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LangChain components
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from src.document_processor import process_documents, DocumentHelper

# Set the title and description
st.title("Document Intelligence System")
st.write("Upload documents and ask questions about their content")

# Add this function after the imports and before initializing session state
def get_available_ollama_models():
    """Fetch available models from Ollama API"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        else:
            return []
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return []

# Initialize session state for storing processed documents
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'document_store' not in st.session_state:
    st.session_state.document_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Chat"
if 'available_models' not in st.session_state:
    st.session_state.available_models = get_available_ollama_models()

# Update the model initialization in session state
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(model="deepseek-r1:latest", base_url="http://localhost:11434")
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
if 'doc_helper' not in st.session_state:
    st.session_state.doc_helper = DocumentHelper()

# Function to chat with documents using LangChain and Ollama
# In the chat_with_documents function, after getting the result:

def chat_with_documents(query, vector_store, stream=False):
    if vector_store is None:
        return "Please process documents first before asking questions."
    
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
    # Define a custom prompt template
    template = """
    You are a helpful assistant that answers questions based on the provided documents.
    
    Context information from documents:
    {context}
    
    Question: {question}
    
    Please provide a detailed and accurate answer based only on the information in the documents.
    If the information is not found in the documents, please say so.
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create an optimized retriever with better search parameters
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5,  # Only return relevant documents
            "fetch_k": 5  # Fetch more candidates initially
        }
    )
    
    # Create the retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_separator": "\n\n"
        }
    )
    
    # Execute the chain
    result = qa_chain({"query": query})
    response = result["result"]
    
    # Handle deepseek model response
    if "deepseek" in st.session_state.llm.model and '</think>' in response:
        response = response.split('</think>')[-1].strip()
    
    return response

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Batch Processing", "About"])
st.session_state.current_page = page

# Model selection
st.sidebar.header("Model Settings")

# Check if we have the required models
required_models = ["deepseek-r1:latest", "nomic-embed-text:latest"]
missing_models = [model for model in required_models if model not in st.session_state.available_models]

if missing_models:
    st.sidebar.warning(f"Missing required models: {', '.join(missing_models)}")
    st.sidebar.markdown("""
    Please download the missing models using Ollama:
    ```bash
    ollama pull deepseek-r1:latest
    ollama pull nomic-embed-text:latest
    ```
    """)
    if st.sidebar.button("Refresh Available Models"):
        st.session_state.available_models = get_available_ollama_models()
        st.rerun()

# Separate model selections for chat and embeddings
chat_model_options = [model for model in st.session_state.available_models if model != "nomic-embed-text:latest"]
if not chat_model_options:
    chat_model_options = ["deepseek-r1:latest"]

chat_model = st.sidebar.selectbox(
    "Select Chat Model",
    chat_model_options,
    index=chat_model_options.index("deepseek-r1:latest") if "deepseek-r1:latest" in chat_model_options else 0
)

embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    ["nomic-embed-text:latest"],
    index=0,
    disabled=True
)

# Update models if changed
if chat_model != st.session_state.llm.model:
    st.session_state.llm = Ollama(model=chat_model, base_url="http://localhost:11434")
    
if embedding_model != st.session_state.embeddings.model:
    st.session_state.embeddings = OllamaEmbeddings(model=embedding_model, base_url="http://localhost:11434")

# Use available models or default list if none found
# model_options = st.session_state.available_models if st.session_state.available_models else ["deepseek-r1:latest", "llama3", "mistral", "gemma3"]
# model_name = st.sidebar.selectbox(
#     "Select Ollama Model",
#     model_options,
#     index=0 if "deepseek-r1:latest" in model_options else 0
# )

# # Update model if changed
# if model_name != st.session_state.llm.model:
#     st.session_state.llm = Ollama(model=model_name, base_url="http://localhost:11434")

# System Status in sidebar
st.sidebar.header("System Status")
if st.sidebar.button("Check System Status"):
    # Check if the necessary modules and models are available
    try:
        # Check if Ollama is available
        response = st.session_state.llm.invoke("Hello")
        st.sidebar.success("System is ready")
    except Exception as e:
        st.sidebar.error(f"System error: {str(e)}")

# File upload section - common to all pages
# File upload section - only show when not in About page
if st.session_state.current_page != "About":
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "csv", "txt"])

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                # Create temporary files for processing
                temp_files = []
                for uploaded_file in uploaded_files:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                        # Write the uploaded file content to the temporary file
                        tmp.write(uploaded_file.getvalue())
                        temp_files.append(tmp.name)
                    
                # Process the documents using your document processor function
                st.session_state.document_store = process_documents(
                    temp_files, 
                    embeddings=st.session_state.embeddings,
                    doc_helper=st.session_state.doc_helper
                )
                
                # Clean up temporary files
                for temp_file in temp_files:
                    os.remove(temp_file)
                    
                st.session_state.documents_processed = True
                st.success("Documents processed successfully!")
                
                # Add download button for processed documents
                if hasattr(st.session_state.document_store, 'docstore'):
                    processed_docs = []
                    for doc_id in st.session_state.document_store.docstore._dict:
                        doc = st.session_state.document_store.docstore._dict[doc_id]
                        processed_docs.append({
                            "doc_id": doc_id,
                            "filename": doc.metadata.get("source", "unknown"),
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
                    
                    docs_json = json.dumps(processed_docs, indent=2)
                    st.download_button(
                        label="Download Processed Documents",
                        data=docs_json,
                        file_name="processed_documents.json",
                        mime="application/json",
                        key="download_docs"
                    )
                    
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

# Page content based on selection
# In the Chat section, update the query handling:
# In the Chat section
if st.session_state.current_page == "Chat":
    # Chat section
    st.header("Chat with Documents")

    # Add download button for chat history
    if st.session_state.chat_history:
        chat_history_json = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button(
            label="Download Chat History",
            data=chat_history_json,
            file_name="chat_history.json",
            mime="application/json",
            key="download_chat"
        )

    # Add streaming option before the query input
    stream = st.checkbox("Enable streaming response", value=True)

    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**AI:** {content}")

    query = st.text_input("Ask a question about your documents")

    if query and st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question")
        elif not st.session_state.documents_processed:
            st.warning("Please process documents first before asking questions")
        else:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("Generating answer..."):
                # Get the response
                answer = chat_with_documents(query, st.session_state.document_store, stream=stream)
                
                if stream:
                    # Display response with streaming effect
                    response_placeholder = st.empty()
                    full_response = ""
                    for char in answer:
                        full_response += char
                        response_placeholder.markdown(f"**AI:** {full_response}")
                        time.sleep(0.01)  # Adjust speed as needed
                else:
                    # Display response without streaming
                    st.markdown(f"**AI:** {answer}")
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Force a rerun to update the chat history display
                st.rerun()

elif st.session_state.current_page == "Batch Processing":
    # Batch processing section
    st.header("Batch Process Queries")

    # Add a file uploader for batch queries
    st.subheader("Upload Queries")
    query_file = st.file_uploader("Upload a text file with questions (one per line)", type=['txt'], key="batch_query_file")

    # Add a text area for manual input
    st.subheader("Or Enter Queries Manually")
    batch_queries = st.text_area("Enter multiple questions (one per line)")

    # Process queries from either source
    if st.button("Process Batch"):
        if not st.session_state.documents_processed:
            st.warning("Please process documents first before asking questions")
        else:
            # Get queries from file if uploaded
            queries = []
            if query_file is not None:
                queries.extend([line.decode('utf-8').strip() for line in query_file.readlines() if line.strip()])
            
            # Add queries from text area
            if batch_queries:
                queries.extend([q.strip() for q in batch_queries.split("\n") if q.strip()])
            
            if not queries:
                st.warning("Please enter at least one question or upload a file")
            else:
                # Show progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process queries with progress tracking
                answers = []
                for i, query in enumerate(queries):
                    status_text.text(f"Processing query {i+1} of {len(queries)}...")
                    
                    # Generate answer for each query
                    answer = chat_with_documents(query, st.session_state.document_store, stream=False)
                    answers.append(answer)
                    
                    # Update progress
                    progress = (i + 1) / len(queries)
                    progress_bar.progress(progress)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.write("### Results")
                for i, (query, answer) in enumerate(zip(queries, answers)):
                    with st.expander(f"Q{i+1}: {query}"):
                        st.write(answer)
                        
                # Add export functionality
                if st.button("Export Results"):
                    # Prepare results in a format suitable for export
                    export_data = [{
                        "question": query,
                        "answer": answer
                    } for query, answer in zip(queries, answers)]
                    
                    # Convert to JSON string
                    json_str = json.dumps(export_data, indent=2)
                    
                    # Create a download button
                    st.download_button(
                        label="Download Results as JSON",
                        data=json_str,
                        file_name="batch_results.json",
                        mime="application/json"
                    )

elif st.session_state.current_page == "About":
    # About section
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
    st.write("tankwin08")
    
    st.write("### License")
    st.write("This project is licensed under the MIT License")