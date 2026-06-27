# 📄 Conversational RAG with PDF & Chat History

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square"/>
  <img src="https://img.shields.io/badge/Groq-F55036?style=flat-square"/>
  <img src="https://img.shields.io/badge/Gemma2--9b-4285F4?style=flat-square&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
</p>

A **conversational RAG application** that lets you upload multiple PDFs and ask questions about them — with full chat history awareness. Built on LangChain, Groq (Gemma2-9b), ChromaDB, and HuggingFace embeddings. The key feature: the system reformulates follow-up questions using prior conversation context before retrieving, so it always retrieves the right chunks even for vague follow-ups like *"tell me more about that"*.

---

## What Makes This Different from Basic RAG

Most RAG demos are **stateless** — every question is treated independently. This app implements **history-aware retrieval**:

```
User: "What is the main finding of the paper?"
AI:   "The main finding is X..."

User: "Can you elaborate on that?"
      ↑
      Without history-awareness: retriever searches for "that" → wrong chunks
      With history-awareness:    LLM first reformulates →
      "Can you elaborate on the main finding X?" → correct chunks retrieved
```

This is achieved with LangChain's `create_history_aware_retriever` — a dedicated LLM call that reformulates the question before retrieval.

---

## Architecture

```
PDF Upload(s)
     │
     ▼
PyPDFLoader → raw documents
     │
     ▼
RecursiveCharacterTextSplitter
  chunk_size=5000, chunk_overlap=500
     │
     ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
     │
     ▼
ChromaDB vector store → retriever
     │
     │         ┌─────────────────────────────────┐
     │         │   CHAIN 1: Question Reformulation│
     ├────────►│   create_history_aware_retriever │
     │         │   llm + retriever + chat_history │
     │         │   → standalone question          │
     │         └──────────────┬──────────────────┘
     │                        │ reformulated question
     │                        ▼
     │         ┌─────────────────────────────────┐
     │         │   CHAIN 2: Answer Generation    │
     │         │   create_stuff_documents_chain  │
     │         │   retrieved chunks + qa_prompt  │
     │         │   → concise answer (≤3 sentences)│
     │         └──────────────┬──────────────────┘
     │                        │
     ▼                        ▼
RunnableWithMessageHistory   answer + updated chat history
  (session-aware wrapper)
```

---

## Two-Chain Design

### Chain 1 — History-Aware Retriever

```python
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

This runs **before** retrieval — it rewrites the user's question into a self-contained query using the chat history, so the retriever always gets a meaningful search term.

### Chain 2 — Question Answering

```python
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

`create_stuff_documents_chain` stuffs all retrieved chunks into the prompt context. The LLM is explicitly instructed to answer in **3 sentences max** and admit when it doesn't know — preventing hallucination.

---

## Session Management

Each conversation is tracked by a `session_id` — stored in `st.session_state.store`:

```python
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
```

Multiple sessions can run simultaneously — each `session_id` gets its own isolated `ChatMessageHistory`. Switch the session ID in the UI to start a fresh conversation without losing the previous one.

---

## Chunking Strategy

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,    # large chunks — preserves paragraph context
    chunk_overlap=500   # 10% overlap — prevents answers from being split across chunks
)
```

`chunk_size=5000` is intentionally large (most tutorials use 1000) — this preserves more context per chunk, which is important for technical PDFs where answers often span multiple paragraphs.

---

## Embeddings & Vector Store

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

- **`all-MiniLM-L6-v2`** — a lightweight but high-quality sentence transformer. 384-dimensional embeddings. Runs locally via HuggingFace — no embedding API cost.
- **ChromaDB** — in-memory vector store, rebuilt fresh on each PDF upload. Fast for interactive use.

---

## LLM — Groq + Gemma2-9b

```python
llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
```

**Groq** provides ultra-fast inference (LPU hardware) — responses are near-instant compared to standard API endpoints. **Gemma2-9b** is Google's open-weight model, strong at instruction-following and document Q&A tasks.

---

## Tech Stack

| Component | Technology | Detail |
|-----------|-----------|--------|
| LLM | Gemma2-9b via Groq | Fast inference, strong Q&A |
| Embeddings | `all-MiniLM-L6-v2` | Local HuggingFace — no API cost |
| Vector store | ChromaDB | In-memory, rebuilt per session |
| PDF loading | `PyPDFLoader` | Multi-page PDF extraction |
| Chunking | `RecursiveCharacterTextSplitter` | 5000 chars, 500 overlap |
| History chain | `create_history_aware_retriever` | Question reformulation before retrieval |
| Session state | `RunnableWithMessageHistory` | Per-session chat history isolation |
| UI | Streamlit | File uploader, chat input, history display |

---

## Setup

```bash
git clone https://github.com/utkarsh027/PDF-QandA-ChatBot
cd PDF-QandA-ChatBot
pip install -r requirements.txt
```

**Create `.env`:**
```env
HF_TOKEN=hf_...
```

**Run:**
```bash
streamlit run app.py
```

---

## Usage

1. Enter your **Groq API key** in the text field (free at console.groq.com)
2. Set a **Session ID** (default: `default_session`) — change this to start a new conversation
3. **Upload one or more PDFs** — multiple files are supported and merged
4. **Ask questions** — follow-up questions work naturally, the system remembers context

---

## Project Structure

```
PDF-QandA-ChatBot/
├── app.py            # All logic: PDF loading, RAG chain, session state, Streamlit UI
├── requirements.txt
├── .env              # HF_TOKEN (not committed)
└── temp.pdf          # Temporary file written during PDF processing (auto-deleted)
```

---

## Author

**Utkarsh Tiwari** — MSc Data Science, UC Irvine  
[github.com/utkarsh027](https://github.com/utkarsh027) · [LinkedIn](https://www.linkedin.com/in/utkarsh-tiwari-330857166/)
