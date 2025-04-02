import streamlit as st
import json

def render_batch_processing(document_store, chat_with_documents_func):
    """Render the batch processing interface"""
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
                    answer = chat_with_documents_func(query, document_store, stream=False)
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
                results_json = json.dumps([{
                    "question": query,
                    "answer": answer
                } for query, answer in zip(queries, answers)], indent=2)
                
                st.download_button(
                    label="Download Results as JSON",
                    data=results_json,
                    file_name="batch_results.json",
                    mime="application/json"
                )