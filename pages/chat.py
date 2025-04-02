import streamlit as st
import json
import time

def render_chat(document_store, llm, chat_with_documents_func):
    """Render the chat interface"""
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
                answer = chat_with_documents_func(query, document_store, stream=stream)
                
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