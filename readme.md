# ⚖️ LawBot – Your Smart Legal Guide

**LawBot** is a smart, multilingual legal assistant powered by Retrieval-Augmented Generation (RAG). It helps users receive concise and context-aware answers to legal questions and provides ready-to-use legal document templates like rental agreements, affidavits, NDAs, and more.

---

## 🚀 Features

- ✅ RAG-based legal question answering from a custom PDF knowledge base
- 🌍 Multilingual output support (auto-detect and translate)
- 📄 Instant access to common legal document templates
- 🧠 Vector search powered by Chroma + HuggingFace embeddings
- 🗨️ Streamlit interface with session chat memory
- 🔐 Secure key handling with `.env` support

---

## 📁 Project Structure

lawbot/


├── data/ # Contains law-related PDF documents
├── legal_templates/ # Contains legal document templates in .txt format
├── data-ingestion.py # Builds Chroma vector DB from PDFs
├── main-app.py # Streamlit chatbot interface
├── .env # Your secret API keys
├── requirements.txt
└── README.md


## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lawbot.git
cd lawbot
2. Create a Virtual Environment
bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

3. Install Dependencies
bash
pip install -r requirements.txt

4. Set Your API Keys
Create a .env file in the root directory:

env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key

5. Prepare Your Data
Add legal PDFs to the data/ folder.

Add legal templates (e.g., Rental_Agreement.txt, Power_of_Attorney.txt) to the legal_templates/ folder.

6. Build the Vector Store
bash
python data-ingestion.py

7. Launch the Application
bash
streamlit run main-app.py
