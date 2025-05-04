# Document Q&A System with Gemini and LangChain

This project is a **Document Question Answering (Q&A) system** that leverages Google Gemini LLM and LangChain to answer questions about the content of a document. It uses Retrieval-Augmented Generation (RAG) to ensure responses are grounded in the actual document.

---

## 🚀 Features

- **PDF Parsing**: Reads and splits PDFs into manageable text chunks.
- **Semantic Search**: Uses Gemini embeddings and FAISS vector store for efficient retrieval.
- **LLM-Powered Q&A**: Answers user questions using Gemini-1.5-Pro, grounded in the uploaded document.
- **Custom Prompt Engineering**: Ensures concise, context-aware, and honest answers.
- **Interactive CLI**: Ask questions in the terminal and get instant responses.

---

## 📚 Tech Stack

- **Python 3.12**
- [LangChain](https://python.langchain.com/)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pdf)

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[PDF] --> B[PyPDFLoader]
    B --> C[CharacterTextSplitter]
    C --> D[Gemini Embeddings]
    D --> E[FAISS Vector Store]
    E --> F[Retriever]
    F --> G[Gemini LLM (Q&A)]
    G --> H[User]
```

- **Document Loading**: PDF is loaded and split into text chunks.
- **Embedding & Indexing**: Chunks are embedded using Gemini and stored in FAISS.
- **Retrieval-Augmented Generation**: Relevant chunks are retrieved and passed to Gemini LLM with a custom prompt for answer generation.

---

## ⚡️ Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/document-qa-system.git
cd document-qa-system
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

- Create a `.env` file in the project root:
  ```
  GOOGLE_API_KEY=your_google_api_key_here
  ```
- **Do NOT share or commit your `.env` file.**

### 4. Add Your PDF

- Place your PDF in the `data/` directory.
- Update the file path in `main.py` if needed.

### 5. Run the Application

```bash
python main.py
```

### 6. Ask Questions

- Type your question in the terminal (e.g., `What is this document about?`).
- Type `exit` to quit.

---

## 📝 Example Prompt

```
Ask a question (type 'exit' to quit): What technical skills are listed?
Answer:
- Python
- Machine Learning
- Data Analysis
```

---

## 🛠️ Customization

- **Prompt Engineering**: Modify the `CUSTOM_PROMPT` in `main.py` to change answer style or constraints.
- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` for different document types.

---

## 🧩 Limitations & Notes

- **Model Name**: Ensure you use the correct Gemini model (`models/gemini-1.5-pro-latest`).  
  If you see a `404 models/gemini-pro is not found` error, update your code to use the correct model name.
- **API Key Security**: Your `.env` file is ignored by Git. Never share your API key publicly.
- **CLI Only**: This version runs in the terminal. For a web interface, consider using [Streamlit](https://streamlit.io/).

---

## 📄 License

This project is for educational and demonstration purposes.

---

## 🙏 Acknowledgements

- [LangChain Documentation](https://python.langchain.com/docs/)
- [Google Generative AI (Gemini) API](https://ai.google.dev/)
- [FAISS by Facebook Research](https://github.com/facebookresearch/faiss)

---

**Feel free to fork, contribute, or reach out with questions!**

---

**Tip:**  
If you encounter errors about model names or API keys, double-check your `.env` file and ensure you are using the latest Gemini model as shown above.

![Screenshot 2025-05-04 213551](https://github.com/user-attachments/assets/4c727930-6d8b-4bbb-a601-43bb500e8d81)

![Screenshot 2025-05-04 213522](https://github.com/user-attachments/assets/4e04804b-92be-43a7-b0a0-25ee24391a63)
