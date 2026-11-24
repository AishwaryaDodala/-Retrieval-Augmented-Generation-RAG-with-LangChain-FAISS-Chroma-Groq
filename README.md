# Retrieval-Augmented-Generation-RAG-with-LangChain-FAISS-Chroma-Groq
A hands-on, step-by-step implementation of a **Retrieval-Augmented Generation (RAG)** pipeline. It covers every building block: document loading, text splitting, embeddings, vector databases and LLM-based answering.


## 🎯 Objective

To understand and build the core components of a RAG system from scratch using LangChain, so that a Large Language Model can answer questions from **external, custom documents** instead of relying only on its training data.

---

## ❓ Problem Definition

Large Language Models have a fixed knowledge cutoff and cannot see private or domain-specific documents. Passing a full long document into the prompt is also inefficient and often exceeds context limits.

**Goal:** Convert raw documents into a searchable knowledge base, retrieve only the most relevant pieces for a query, and let the LLM generate answers grounded in that retrieved context.

---

## 📂 Dataset Used

| Source | Type | Used For |
|---|---|---|
| `state_of_the_union.txt` | Long-form speech text (~39 KB) | Main RAG pipeline: loading, chunking, embedding, FAISS indexing |
| `game.txt` | Cricket poem (~8 KB) | Document loader + LLM summarization chain |
| Custom cricket player documents (7 IPL players: Dhoni, Rohit, Bumrah, Kohli, Abhishek Sharma, Pant, Ishan Kishan) | Hand-written `Document` objects with `team` metadata | Chroma vector DB: similarity search, update and delete |
| Product web page (via `WebBaseLoader`) | Web page content | Question answering over web-based data |
| Short space-exploration paragraph | Sample text | Comparing text splitters |

---

## 🧪 Methodology

The project is organised as five notebooks, each covering one stage of the pipeline:

| # | Notebook | What it covers |
|---|---|---|
| 1 | `document_loader.ipynb` | Loading `.txt` files with `TextLoader`; building a prompt → LLM → parser chain to summarize a document; overview of `PyPDFLoader`, `CSVLoader`, `WebBaseLoader` |
| 2 | `WebBased_Document.ipynb` | Loading web pages with `WebBaseLoader` and answering user questions from the page content |
| 3 | `Text_Splitter_58.ipynb` | `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter`; effect of `chunk_size` and `chunk_overlap` |
| 4 | `Vector_database.ipynb` | Creating embeddings, storing documents with metadata in **ChromaDB**, similarity search (`k` = 1, 2, 3), updating and deleting records |
| 5 | `RAG_Implementation_58.ipynb` | End-to-end pipeline: load → split → embed → index in **FAISS** |

### Pipeline flow

```
Documents → Loader → Text Splitter → Embeddings → Vector Store → Retriever → LLM → Answer
```

1. **Load:** `TextLoader` reads the document.
2. **Split:** `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=0`) breaks it into chunks.
3. **Embed:** `sentence-transformers/all-MiniLM-L6-v2` converts chunks into dense vectors.
4. **Index:** Vectors are stored in FAISS (and ChromaDB, with persistence, in the vector DB notebook).
5. **Retrieve:** Semantic similarity search returns the top-k relevant chunks.
6. **Generate:** The retrieved context goes into a prompt template, and the LLM (`openai/gpt-oss-120b` via Groq) produces the answer.

---

## 🛠️ Technology Used

- **Language:** Python
- **Environment:** Google Colab / Jupyter Notebook
- **Framework:** LangChain (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`)
- **LLM:** `openai/gpt-oss-120b` served through **Groq** (`langchain-groq`)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Databases:** FAISS (`faiss-cpu`), ChromaDB
- **Loaders:** `TextLoader`, `WebBaseLoader` (plus `PyPDFLoader` and `CSVLoader` explored)

---

## 📏 Evaluation Metrics

This project is a learning-focused implementation, so the pipeline was evaluated **qualitatively** rather than with benchmark scores:

- **Retrieval relevance:** For queries such as *"who among these are bowler"* and *"Wicketkeeper batsman?"*, the top-k results were checked manually to confirm the correct player documents came back.
- **Chunk quality:** Splitter outputs were inspected for chunk length and for whether sentences stayed coherent, comparing character-based and recursive splitting and different overlap values.
- **Answer grounding:** LLM answers were checked against the source document or web page content.
- **Data operations:** Add, update and delete operations in ChromaDB were verified by inspecting the stored records.

> 🔭 **Future work:** quantitative RAG evaluation with Precision@k / Recall@k / MRR for retrieval, and faithfulness and answer relevancy (for example with RAGAS) for generation.

---

## 💡 Key Insights

- **Chunking strategy matters:** `RecursiveCharacterTextSplitter` keeps text coherent by splitting on paragraphs, then sentences, then words, whereas `CharacterTextSplitter` can cut text mid-sentence.
- **Chunk overlap preserves context** between neighbouring chunks, at the cost of some redundancy.
- **Semantic search beats keyword search:** a query like *"who among these are bowler"* retrieves the right player even without an exact keyword match.
- **Metadata** (such as `team`) attached to documents enables richer filtering and organisation.
- **Vector DBs are manageable like regular databases:** records can be added, updated and deleted. ChromaDB adds persistence and collections, while FAISS is lightweight and fast for in-memory search.
- **RAG reduces hallucination** by grounding the LLM in retrieved context instead of relying on its memory alone.

---

## ✅ Conclusion

This project walks through the complete RAG workflow, from raw text and web pages to a searchable vector index that an LLM can query. It shows how each component (loader, splitter, embedding model, vector store, LLM) affects retrieval quality and answer grounding.

**Possible extensions:** add a retriever + generation chain with a QA prompt, support PDFs, add re-ranking and hybrid search, build a Streamlit/Gradio UI, and add quantitative evaluation.

---



---

## 📁 Repository Structure

```
├── document_loader.ipynb
├── WebBased_Document.ipynb
├── Text_Splitter_58.ipynb
├── Vector_database.ipynb
├── RAG_Implementation_58.ipynb
├── state_of_the_union.txt
├── game.txt
└── README.md
```
