# Document Q&A Agent — RAG

A Python-based Document Question Answering Agent that allows users to ask questions about PDF documents and receive context-grounded answers using Retrieval-Augmented Generation (RAG).

## Features

- Extracts text from PDF documents
- Splits text into manageable chunks
- Generates semantic embeddings using Sentence Transformers
- Retrieves relevant chunks using FAISS similarity search
- Uses the Gemini API to generate answers based on retrieved context
- Provides a foundation for document-based question answering

## Tech Stack

- **Language:** Python
- **LLM:** Google Gemini API
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search:** FAISS
- **PDF Processing:** pypdf
- **Text Splitting:** LangChain Text Splitters
- **Interface:** Streamlit (if used in the current app)

## How It Works

1. Load and extract text from a PDF.
2. Split the extracted text into overlapping chunks.
3. Convert chunks into vector embeddings.
4. Store the embeddings in a FAISS index.
5. Embed the user's question and retrieve relevant chunks.
6. Pass the retrieved context and question to Gemini.
7. Return an answer grounded in the document content.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/document-qa-agent.git
cd document-qa-agent
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own Gemini API key. Never commit your `.env` file.

### 5. Run the application

Use the command appropriate for the entry point in your repository:

```powershell
streamlit run app.py
```

## Security

Keep API keys and private documents out of version control. Store credentials in environment variables.

## Future Improvements

- Support multiple PDF uploads
- Display source chunks alongside answers
- Improve retrieval evaluation
- Add conversation history and a polished user interface
- Deploy the application online

## Author

**Adapa Nishith Reddy**

GitHub: https://github.com/nishith232
