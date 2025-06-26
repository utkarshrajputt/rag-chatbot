[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkarshrajputt/rag-chatbot/blob/main/RAG2.ipynb)

# 📚 RAG Chatbot: Ask Docs Anything

A powerful Retrieval-Augmented Generation (RAG) chatbot built using:
- 🧠 Hugging Face Transformers (Mistral 7B / Phi-2)
- 🔍 Semantic Search with FAISS + Sentence Transformers
- 📄 PDF & DOCX ingestion + text chunking
- 💬 Streamlit frontend with file upload & answer display
- ☁️ Hosted via ngrok on Google Colab

---

## 🚀 Features
- Upload any PDF or DOCX
- Ask questions in natural language
- Get precise, context-aware answers
- Lightweight backend (Colab + GPU = 💸 saved)
- No hallucinations — just 🔥 RAG power

---

## 🔧 Tech Stack
| Tool | Usage |
|------|-------|
| `transformers` | LLM loading (Mistral-7B / Phi-2) |
| `sentence-transformers` | Semantic embeddings |
| `faiss` | Vector similarity search |
| `PyPDF2`, `python-docx` | File parsing |
| `nltk` | Text chunking |
| `streamlit` | UI |
| `ngrok` | Share your app online |

---

## ⚙️ Run It (Colab + Streamlit)
1. Open the [Colab Notebook](./RAG2.ipynb)
2. Upload your PDF or DOCX
3. Ask anything from it
4. 🔗 Share the Streamlit URL via ngrok

---

## 🤖 Sample Questions
> "Summarize the document."  
> "What are the key findings in section 2?"  
> "List all deadlines mentioned."

---

## 📦 Setup
```bash
pip install -r requirements.txt
# OR
pip install torch transformers sentence-transformers faiss-cpu PyPDF2 python-docx streamlit




---

## 🤝 Contributing

Pull requests are welcome! If you have ideas to improve this chatbot (like UI, model switching, or memory optimization), feel free to open an issue or submit a PR.

---

