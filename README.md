# 🗺️ GIS Document Assistant

### ITI - GIS Track | Gen AI Course Lab Project
A comprehensive Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, and **Google Gemini** to chat with single or multiple GIS documents (Standards, Manuals, Tutorials).

---

## 🚀 Features

### 📋 Core Features
- **Multi-PDF Upload:** Upload one or more GIS-related PDF documents simultaneously.
- **Intelligent Chunking:** Processes documents using `RecursiveCharacterTextSplitter` to handle structural text layout.
- **Vector Storage:** Stores text embeddings locally using **ChromaDB**.
- **Contextual Q&A:** Uses `gemini-2.5-flash` to answer spatial and GIS questions precisely based only on your data.
- **Source Attributions:** Displays exactly which document pages were referenced to construct the answer.

### 🏆 Bonus Features Implemented
- **Arabic & English Support:** Fully multilingual system that responds in the language of the query.
- **Live Statistics:** Tracks total questions asked and the most frequently cited pages.
- **Session Export:** Download full chat history as a structured `JSON` file.
- **ITI Brand Identity:** UI styled with clean typography and custom thematic elements.

---

## 🔧 Installation & Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd gis-document-assistant
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Or install directly: `pip install streamlit langchain-google-genai langchain-chroma langchain-community langchain-text-splitters pypdf chromadb`)*

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory (see format below) and add your Gemini API Key.

---

## 🏃 How to Run

Launch the Streamlit application by running:
```bash
streamlit run app.py
```

---

## 📂 Project Structure
- `app.py` - The main Streamlit application code.
- `README.md` - Documentation and setup guide.
- `.env` - Local environment secrets.
- `.gitignore` - Specifies intentionally untracked files to ignore.
