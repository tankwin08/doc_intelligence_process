import streamlit as st
import json
import os
from io import BytesIO
import tempfile
import sys

# Add the parent directory to the path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LangChain components
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from src.document_processor import process_documents, DocumentHelper

# Set the title and description
st.title("Document Intelligence System")
st.write("Upload documents and ask questions about their content")

# Initialize session state for storing processed documents
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'document_store' not in st.session_state:
    st.session_state.document_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(model="deepseek-r1:latest", base_url="http://localhost:11434")
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
if 'doc_helper' not in st.session_state:
    st.session_state.doc_helper = DocumentHelper()

# Function to chat with documents using LangChain and Ollama
# In the chat_with_documents function, after getting the result:

def chat_with_documents(query, vector_store, stream=False):
    if vector_store is None:
        return "Please process documents first before asking questions."
    
    # Create a retrieval chain
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
    # Define a custom prompt template
    template = """
    You are a helpful assistant that answers questions based on the provided documents.
    
    Context information from documents:
    {context}
    
    Question: {question}
    
    Please provide a detailed and accurate answer based only on the information in the documents.
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create the retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Execute the chain
    if stream:
        # Note: Streaming might not be directly supported in this way with RetrievalQA
        # This is a simplified approach
        result = qa_chain({"query": query})
        
        # Capture thinking process if using deepseek model
        if "deepseek" in st.session_state.llm.model:
            thinking = "Retrieved documents:\n"
            for i, doc in enumerate(result.get("source_documents", [])):
                thinking += f"\nDocument {i+1}:\n{doc.page_content[:300]}...\n"
            st.session_state.doc_helper.capture_thinking(thinking)
            
        yield result["result"].split('<think>')[1]
    else:
        result = qa_chain({"query": query})
        return result["result"].split('<think>')[1]

# File upload section
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
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# Model selection
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Select Ollama Model",
    ["deepseek-r1:latest", "llama3", "mistral", "gemma3"],
    index=0
)

# Update model if changed
if model_name != st.session_state.llm.model:
    st.session_state.llm = Ollama(model=model_name, base_url="http://localhost:11434")

# Chat section
st.header("Chat with Documents")

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
        
        # Create a placeholder for the streaming response
        response_placeholder = st.empty()
        stream = st.checkbox("Enable streaming response", value=True)
        
        with st.spinner("Generating answer..."):
            if stream:
                # Stream the response
                full_response = ""
                
                # Use a generator function for streaming
                for text_chunk in chat_with_documents(query, st.session_state.document_store, stream=True):
                    full_response += text_chunk
                    # Update the response in real-time
                    response_placeholder.markdown(f"**AI:** {full_response}")
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                # Force a rerun to update the chat history display
                st.rerun()
            else:
                # Regular non-streaming response
                answer = chat_with_documents(query, st.session_state.document_store, stream=False)
                st.markdown(f"**AI:** {answer}")
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Force a rerun to update the chat history display
                st.rerun()

# Batch processing section
st.header("Batch Process Queries")

# Add a file uploader for batch queries
st.subheader("Upload Queries")
query_file = st.file_uploader("Upload a text file with questions (one per line)", type=['txt'])

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

# Add a sidebar with information
with st.sidebar:
    st.header("About")
    st.write("This application allows you to upload documents and ask questions about their content using AI.")
    st.write("Supported file types: PDF, CSV, TXT")
    
    st.header("How to use")
    st.write("1. Upload one or more documents")
    st.write("2. Process the documents")
    st.write("3. Ask questions about the content")
    
    st.header("System Status")
    if st.button("Check System Status"):
        # Check if the necessary modules and models are available
        try:
            # Check if Ollama is available
            response = st.session_state.llm.invoke("Hello")
            st.success("System is ready")
        except Exception as e:
            st.error(f"System error: {str(e)}")