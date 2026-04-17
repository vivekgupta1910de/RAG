# Project: Automotive RAG Assistant

## 🎯 Objective
Build a Retrieval-Augmented Generation (RAG) system to assist automotive engineers in debugging, diagnostics, and documentation search.

The system should:
- Retrieve relevant automotive data (CAN logs, ECU docs, manuals)
- Generate accurate, context-aware answers
- Reduce debugging and analysis time

---

## 🧱 Tech Stack
- Language: Python
- Framework: LangChain
- Vector Database: FAISS
- LLM: OpenAI API
- UI (optional): Streamlit
- Environment: GitHub Codespaces

---

## 📂 Project Structure

rag-project/
│
├── data/                  # Input documents (PDFs, logs, manuals)
├── src/
│   ├── app.py             # Entry point
│   ├── loader.py          # Document loading logic
│   ├── splitter.py        # Text chunking
│   ├── vector_store.py    # FAISS setup and storage
│   ├── query_engine.py    # Retrieval + LLM logic
│
├── requirements.txt
├── .env
└── AGENTS.md

---

## ⚙️ Functional Requirements

### 1. Document Ingestion
- Load PDFs and text files from `/data`
- Support automotive documents (manuals, logs, configs)

### 2. Text Processing
- Split documents into chunks (500–1000 tokens)
- Maintain overlap for context continuity

### 3. Embedding & Storage
- Use OpenAI embeddings
- Store vectors in FAISS
- Enable fast similarity search

### 4. Query System
- Accept user query input
- Retrieve top relevant documents
- Generate answer using LLM

### 5. Output
- Clear, structured responses
- Include context-based answers only (no hallucination)

---

## 🚗 Automotive Use Cases

- CAN communication debugging
- ECU fault analysis
- UDS protocol explanation
- AUTOSAR configuration help
- Infineon TC3xx related issues

---

## 🧪 Coding Guidelines

- Use modular design (separate files for each component)
- Write reusable functions
- Add comments for clarity
- Avoid hardcoding values
- Follow clean code practices

---

## 🔐 Security Rules

- Never hardcode API keys
- Use `.env` for sensitive data
- Validate user input
- Avoid exposing internal data unnecessarily

---

## 📈 Performance Considerations

- Keep chunk size optimized
- Limit number of retrieved documents (top_k = 3–5)
- Avoid unnecessary API calls

---

## 🧠 Prompt Guidelines

- Always use retrieved context for answering
- Do not generate answers without context
- Keep responses concise and technical

---

## 🧪 Testing

- Test with sample automotive queries:
  - "Why does CAN stop when debugger is attached?"
  - "What is UDS security access?"
- Validate accuracy of retrieved documents

---

## 🚀 Future Enhancements

- Add Streamlit UI
- Support real-time CAN log ingestion
- Integrate with MongoDB for history
- Add feedback loop for answer improvement

---

## 📌 Instructions for Agent (Codex)

- Follow project structure strictly
- Do not create unnecessary files
- Keep implementation simple and modular
- Prioritize correctness over complexity
- Ensure code is runnable and tested