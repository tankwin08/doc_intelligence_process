# Document Intelligence System

A powerful document analysis and question-answering system built with Streamlit, LangChain, and Ollama. This application enables users to interact with their documents through natural language queries, leveraging advanced RAG (Retrieval Augmented Generation) technology.

## Features

- 📄 Multi-format Document Support (PDF, CSV, TXT)
- 💬 Interactive Chat Interface
- 🔄 Batch Query Processing
- 📊 Document Processing with Advanced RAG
- 🚀 Optimized Retrieval System
- 📥 Exportable Results in JSON Format
- 🔄 Real-time Streaming Responses
- 🎯 Context-aware Document Analysis

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- Ollama (running locally or on a remote server)
- FAISS for vector storage

## Installation

1. Clone the repository:
```
git clone https://github.com/tankwin08/doc_intelligence_process.git
cd doc_intelligence_process
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Start the Streamlit app:
```
streamlit run streamlit_app.py
```
4. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501 )

## Supported Document Types
- PDF (.pdf)
- CSV (.csv)
- Text (.txt)
- Word Documents (.docx, .doc)
- PowerPoint (.ppt, .pptx)
- Images (.jpg, .jpeg, .png)
## Models
The application uses Ollama to run LLMs locally. By default, it uses:

- deepseek-r1:latest for text generation
- nomic-embed-text for embeddings
You can change the model in the sidebar of the application.


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.