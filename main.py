from fastapi import FastAPI, UploadFile, File, HTTPException
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from dotenv import load_dotenv
import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
import faiss
import numpy as np
import pickle
import google.generativeai as genai
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

load_dotenv()

app = FastAPI()

# Load environment variables
connection_string = os.getenv("BLOB_CONNECTION_STRING")
container_name = os.getenv("BLOB_CONTAINER_NAME")
doc_intel_endpoint = os.getenv("DOC_INTEL_ENDPOINT")
doc_intel_key = os.getenv("DOC_INTEL_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini only if key exists
model = None
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

# Create Blob client only if connection string exists
blob_service_client = None
if connection_string:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create Document Intelligence client only if credentials exist
doc_client = None
if doc_intel_endpoint and doc_intel_key:
    doc_client = DocumentAnalysisClient(
        endpoint=doc_intel_endpoint,
        credential=AzureKeyCredential(doc_intel_key)
    )


@app.get("/")
def home():
    logger.info("Home log")
    return {"message": "FinBot v2 is running"}


@app.get("/version")
def version():
    logger.info("version checked")
    return {"version": "2.0- azure is cool"}



logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string=os.getenv("APPINSIGHTS_CONNECTION_STRING")))
logger.setLevel(logging.INFO)

@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

@app.get("/stock/{symbol}")
def get_stock(symbol: str):
    logger.info(f"Stock requested: {symbol}")
    return {
        "symbol": symbol.upper(),
        "price": 100 + len(symbol),
        "currency": "USD"
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not blob_service_client or not container_name:
        raise HTTPException(status_code=500, detail="Blob service not configured")

    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=file.filename
    )

    file_data = await file.read()
    blob_client.upload_blob(file_data, overwrite=True)

    logger.info("File uploaded via /upload")

    return {"message": f"{file.filename} uploaded successfully"}


@app.post("/extract/{filename}")
async def extract_document(filename: str):
    if not blob_service_client or not container_name:
        raise HTTPException(status_code=500, detail="Blob service not configured")
    if not doc_client:
        raise HTTPException(status_code=500, detail="Document Intelligence not configured")

    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=filename
    )

    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=filename,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=10)
    )

    blob_url = f"{blob_client.url}?{sas_token}"

    poller = doc_client.begin_analyze_document_from_url(
        "prebuilt-layout",
        blob_url
    )

    result = poller.result()

    full_text = ""
    for page in result.pages:
        for line in page.lines:
            full_text += line.content + "\n"

    chunk_size = 1000
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    return {
        "total_characters": len(full_text),
        "total_chunks": len(chunks),
        "sample_chunk": chunks[0] if chunks else ""
    }


@app.post("/index/{filename}")
async def index_document(filename: str):
    if not blob_service_client or not container_name:
        raise HTTPException(status_code=500, detail="Blob service not configured")
    if not doc_client:
        raise HTTPException(status_code=500, detail="Document Intelligence not configured")
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=filename
    )

    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=filename,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=10)
    )

    blob_url = f"{blob_client.url}?{sas_token}"

    poller = doc_client.begin_analyze_document_from_url(
        "prebuilt-layout",
        blob_url
    )

    result = poller.result()

    full_text = ""
    for page in result.pages:
        for line in page.lines:
            full_text += line.content + "\n"

    chunk_size = 1000
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    vectors = []
    for chunk in chunks:
        embedding = genai.embed_content(
            model="gemini-embedding-001",
            content=chunk
        )
        vectors.append(embedding["embedding"])

    vectors_np = np.array(vectors).astype("float32")
    dimension = vectors_np.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    faiss.write_index(index, "faiss_index.index")

    with open("metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return {
        "total_chunks": len(chunks),
        "vector_dimension": dimension
    }


@app.post("/ask")
async def ask_question(question: str):
    if not gemini_api_key or not model:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    if not os.path.exists("faiss_index.index") or not os.path.exists("metadata.pkl"):
        raise HTTPException(status_code=500, detail="Index files not found. Run /index first.")

    index = faiss.read_index("faiss_index.index")

    with open("metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    query_embedding = genai.embed_content(
        model="gemini-embedding-001",
        content=question
    )["embedding"]

    query_vector = np.array([query_embedding]).astype("float32")

    k = min(5, len(chunks))
    distances, indices = index.search(query_vector, k)

    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a financial analyst assistant for HDFC Bank.

Use ONLY the provided context to answer.
If comparing quarters, compute differences clearly.
If answer not found, say 'Information not available in provided reports.'

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(prompt)

    logger.info("Query retrieved via LLM")

    return {
        "answer": response.text,
        "retrieved_chunks_count": len(retrieved_chunks)
    }