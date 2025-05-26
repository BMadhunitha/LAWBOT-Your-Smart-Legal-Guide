import os
import time
from dotenv import load_dotenv

# LangChain dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_chroma import Chroma

# Load env vars
load_dotenv()

# Directories setup
current_dir_path = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
data_path = os.path.join(current_dir_path, "data")  # Folder with PDF files
persistent_directory = os.path.join(current_dir_path, "data-ingestion-local")  # Vector DB folder

if not os.path.exists(persistent_directory):
    print("[INFO] Initiating the build of Vector Database .. 📌📌\n")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ALERT] {data_path} doesn't exist. ⚠️⚠️")

    # List all PDFs in data folder
    pdfs = [pdf for pdf in os.listdir(data_path) if pdf.endswith(".pdf")]

    doc_container = []

    # Load each PDF and collect documents
    for pdf in pdfs:
        loader = PyPDFLoader(file_path=os.path.join(data_path, pdf), extract_images=False)
        docsRaw = loader.load()
        doc_container.extend(docsRaw)

    # Split documents into chunks (1000 chars with 50 overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs_split = splitter.split_documents(doc_container)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs_split)}\n")

    # Create embeddings
    embedF = CohereEmbeddings(model="embed-english-light-v3.0",user_agent="my-legal-app/1.0")
    print("[INFO] Started embedding")
    start = time.time()

    # Build and persist vector DB
    vectorDB = Chroma.from_documents(
        documents=docs_split,
        embedding=embedF,
        persist_directory=persistent_directory
    )

    end = time.time()
    print("[INFO] Finished embedding")
    print(f"[ADD. INFO] Time taken: {end - start} seconds")

else:
    print("[ALERT] Vector Database already exists. ️⚠️")
