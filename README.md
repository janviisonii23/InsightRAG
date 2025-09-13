# RAGpedia - A RAG Plugin for Document-Based Q&A

A production-ready plugin that transforms your company's PDFs and DOCX documents into a **queryable knowledge assistant**. It uses **Retrieval-Augmented Generation (RAG)** to deliver intelligent, context-aware responses — complete with **images**, **tables**, and **code snippets**, all formatted in **Markdown** so you can render them directly in your frontend.

---

## Ideal For

-  Internal knowledge bases
-  Customer support automation
-  Financial software FAQs (e.g., Tally, Zoho Books)
-  Onboarding and policy guidance
-  Developer documentation bots

---

##  Folder Structure

```
RAGpedia/
├── app.py                        # FastAPI API server
├── main.py                       # Script to test functionality without frontend
├── requirements.txt              # Python dependencies
├── .env                          # API key and environment config
├── data/                         # Stores uploaded files and embeddings
│
└── src/
    ├── extraction/
    │   └── unstructured_extraction.py
    ├── chunking/
    │   └── chunker.py
    ├── embedding/
    │   └── chroma_embedder.py
    └── rag_pipeline/
        ├── context_builder.py
        ├── llm_wrapper.py
        ├── query_embedder.py
        └── retriever.py

```

---

##  Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/Sohamshah03/RAGpedia.git
cd RAGpedia
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Install Poppler (For PDF image extraction)

`poppler` is required by the `unstructured` library to extract images and tables from PDFs.

- **Windows**:  
  Download binaries from [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases)  
  Add the `/bin` folder to your system **PATH**.

- **macOS**:

```bash
brew install poppler
```

- **Linux**:

```bash
sudo apt install poppler-utils
```

---

## Environment Variables

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=sk-xxxxx
```

---

## Running the Backend

```bash
uvicorn app:app --reload
```

---

## Testing Locally

```bash
python main.py
```

---

## API Flow

- **POST** `/new-session` → Create a new session ID
- **POST** `/upload` → Upload PDF/DOCX and process
- **POST** `/query` → Ask questions about the uploaded document

---

## Sample Response (Markdown)

```markdown
### 1. Steps to Generate Invoice in Tally

- Go to **Gateway of Tally → Accounting Vouchers**
- Press `F8` to select **Sales**
- Fill in party details and item info

![Invoice Screenshot](data/session_abcd/images/invoice.png)

### 2. Notes

- Ensure GST Number is configured
```

> The backend returns Markdown — you can plug this into any dashboard or chatbot frontend to render rich responses with images, tables, and code snippets.

---

## Architecture Highlights

- Uses `unstructured` to extract rich elements: text, images, tables, code
- Chunks documents using a custom `StructuredChunker`
- Stores embeddings per session in **ChromaDB**
- Leverages **OpenRouter** (Mistral 7B) to generate responses
- Returns clean **Markdown** for flexible rendering

---

## Flexibility

- Frontend-agnostic: use with any React/Vue dashboard, bot, or mobile app
- Returns rich Markdown responses, including images, tables and code snippets
- Can be used as a plugin in enterprise/internal tooling

---

## License

MIT License — use it, fork it, build on it!

---

## Built By

**Soham Shah**  
[LinkedIn](https://linkedin.com/in/sohamshah03) • [GitHub](https://github.com/Sohamshah03)
