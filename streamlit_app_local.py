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
from pages.chat import render_chat
from pages.batch_processing import render_batch_processing
from pages.about import render_about

# Set the page config with a custom menu
st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide the default Streamlit pages navigation
hide_pages_navigation = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stSidebarNav"] {display: none;}

/* Make Navigation text and radio buttons larger */
div.row-widget.stRadio > div {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}
div.row-widget.stRadio > div[role="radiogroup"] > label {
    font-size: 1.1rem !important;
    padding: 5px 0px;
}
.sidebar .sidebar-content .block-container h1, 
.sidebar .sidebar-content .block-container h2, 
.sidebar .sidebar-content .block-container h3 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* Enhanced Banner styling with matching background color */
.banner-container {
    background-color: #0066b2; /* Adjust this color to match your logo */
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.banner-text {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
}
.banner-subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
    margin-top: 5px;
}
</style>
"""
st.markdown(hide_pages_navigation, unsafe_allow_html=True)

# After the style definitions but before using the function

# Add a function to encode the image as base64 for inline HTML display
def get_base64_encoded_image(image_path):
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Create a banner with logo and title with matching background color
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "mic_log.png")
if os.path.exists(logo_path):
    # Create a custom banner with HTML
    st.markdown(
        f"""
        <div class="banner-container">
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{get_base64_encoded_image(logo_path)}" width="100" style="margin-right: 20px;">
                <div>
                    <h1 class="banner-text">Document Intelligence System</h1>
                    <p class="banner-subtitle">Upload documents and ask questions about their content</p>
                </div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
else:
    # Fallback if logo doesn't exist
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
# st.sidebar.title("Document Intelligence System")  # Move the title to sidebar
st.sidebar.markdown("<h2 style='font-size: 2.5rem; font-weight: 700;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Chat", "Batch Processing", "About"])
st.session_state.current_page = page

# Model selection
st.sidebar.markdown("<h2 style='font-size: 1.5rem; font-weight: 700;'>Model Settings</h2>", unsafe_allow_html=True)

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

# System Status in sidebar
st.sidebar.markdown("<h2 style='font-size: 1.5rem; font-weight: 700;'>System Status</h2>", unsafe_allow_html=True)
if st.sidebar.button("Check System Status"):
    # Check if the necessary modules and models are available
    try:
        # Check if Ollama is available
        response = st.session_state.llm.invoke("Hello")
        st.sidebar.success("System is ready")
    except Exception as e:
        st.sidebar.error(f"System error: {str(e)}")

# File upload section - only show when not in About page
if st.session_state.current_page != "About":
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "csv", "txt","docx","pptx","jpg"])

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
if st.session_state.current_page == "Chat":
    render_chat(st.session_state.document_store, st.session_state.llm, chat_with_documents)
elif st.session_state.current_page == "Batch Processing":
    render_batch_processing(st.session_state.document_store, chat_with_documents)
elif st.session_state.current_page == "About":
    render_about()