from fastapi import FastAPI, UploadFile, File
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
import faiss
import numpy as np
import pickle


load_dotenv()

app = FastAPI()
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Load environment variables
connection_string = os.getenv("BLOB_CONNECTION_STRING")
container_name = os.getenv("BLOB_CONTAINER_NAME")

# Create Blob client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

doc_client = DocumentAnalysisClient(
    endpoint=os.getenv("DOC_INTEL_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("DOC_INTEL_KEY"))
)
@app.get("/")
def home():
    return {"message": "FinBot v2 is running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=file.filename
    )

    file_data = await file.read()
    blob_client.upload_blob(file_data, overwrite=True)

    return {"message": f"{file.filename} uploaded successfully"}

@app.post("/extract/{filename}")
async def extract_document(filename: str):

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

    # simple chunking (temporary)
    chunk_size = 1000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    return {
        "total_characters": len(full_text),
        "total_chunks": len(chunks),
        "sample_chunk": chunks[0]
    }

@app.post("/index/{filename}")
async def index_document(filename: str):

    # ----- Generate SAS URL -----
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

    # ----- Extract Document -----
    poller = doc_client.begin_analyze_document_from_url(
        "prebuilt-layout",
        blob_url
    )

    result = poller.result()

    full_text = ""
    for page in result.pages:
        for line in page.lines:
            full_text += line.content + "\n"

    # ----- Chunking -----
    chunk_size = 1000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    # ----- Generate Embeddings -----
    vectors = []

    for chunk in chunks:
        embedding = genai.embed_content(
            model="gemini-embedding-001",
            content=chunk
        )
        vectors.append(embedding["embedding"])

    vectors_np = np.array(vectors).astype("float32")

    dimension = vectors_np.shape[1]

    # ----- Create FAISS Index -----
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    # ----- Save Index + Metadata -----
    faiss.write_index(index, "faiss_index.index")

    with open("metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return {
        "total_chunks": len(chunks),
        "vector_dimension": dimension
    }

@app.post("/ask")
async def ask_question(question: str):

    # ----- Load FAISS index -----
    index = faiss.read_index("faiss_index.index")

    with open("metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    # ----- Embed Question -----
    query_embedding = genai.embed_content(
        model="gemini-embedding-001",
        content=question
    )["embedding"]

    query_vector = np.array([query_embedding]).astype("float32")

    # ----- Retrieve Top-5 -----
    k = 5
    distances, indices = index.search(query_vector, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    # ----- Build Financial Prompt -----
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

    # ----- Generate Answer -----
    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "retrieved_chunks_count": len(retrieved_chunks)
    }

@app.get("/version")
def version():
    return {"version": "1.0"}