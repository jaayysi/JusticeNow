# 📄 LangChain PDF Chat using RAG

A beginner-friendly Streamlit application for **chatting with PDF files** using **LangChain**, **FAISS**, and **HuggingFace**. This app demonstrates the basic principles of **Retrieval-Augmented Generation (RAG)** in an interactive and educational way.

---

## 💡 What It Does
- ✅ Accepts a **PDF upload**
- 🤖 Uses **embeddings + LLM** to answer questions about the document
- ⚡ Returns **grounded, accurate responses** based on the uploaded content

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:
- A **retriever** (fetches relevant document chunks)
- A **generator** (uses a language model to generate the final answer)

This helps the AI give more **factually correct answers** by "looking up" information before generating a response.

---

## 🛠️ Technologies Used

| Library | Purpose |
|--------|---------|
| [Streamlit](https://streamlit.io) | Web UI |
| [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/) | Extracts text from PDFs |
| [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) | Loads pre-trained language models |
| [LangChain](https://www.langchain.com/) | RAG pipeline and abstraction layer |
| [FAISS](https://github.com/facebookresearch/faiss) | Fast vector similarity search |
| [dotenv](https://pypi.org/project/python-dotenv/) | Manages API keys and environment variables |

---

## 🧾 How It Works

### 1. Upload a PDF

The app reads the document and extracts all text using `PyMuPDF`.

### 2. Split into Chunks

The text is split into overlapping chunks using `RecursiveCharacterTextSplitter`.
```python
chunk_size=500
chunk_overlap=50
````

### 3. Convert Chunks to Embeddings

The app uses `all-MiniLM-L6-v2` to convert text into vector embeddings.

### 4. Store in FAISS Vector Store

Embeddings are saved in an FAISS index for similarity-based retrieval.

### 5. Ask Questions

The user types a question. The app:

* Retrieves the most relevant chunks (`k=2`)
* Sends them with the question to an LLM (`flan-t5-base`)
* Displays the AI-generated answer

---

## 📦 Installation

```
git clone https://github.com/jlrn-rph/ama-prompt-rag-template
cd langchain-pdf-chat
pip install -r requirements.txt
```

Make sure to add your Hugging Face API key (if using hosted models):

```env
# .env
HF_API_KEY=your_api_key_here
```


## 🚀 Running the App

```
streamlit run app.py
```

Then, open `http://localhost:8501` in your browser.

---

## 🔄 Example Use Case

* Upload a **research paper**
* Ask: *“What are the main findings of this paper?”*
* Get a summarized answer sourced directly from the document

---

## 📚 Learning Resources
* [LangChain Docs](https://docs.langchain.com/)
* [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
* [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
* [Streamlit Docs](https://docs.streamlit.io/)
* [Intro to RAG (Blog)](https://www.pinecone.io/learn/retrieval-augmented-generation/)
