import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os

def load_incident_data(filepath: str) -> list[Document]:
    """
    Lê o CSV de incidentes e transforma cada linha em um Document do LangChain.
    """
    df = pd.read_csv(filepath)
    documents = []
    for _, row in df.iterrows():
        text = f"Incident type: {row['incident_type']}. Description: {row['description']}."
        metadata = {"order_id": row['order_id'], "reported_at": row['reported_at']}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def ingest_documents(documents: list[Document], persist_directory: str = "vectorstore"):
    """
    Gera embeddings e armazena os documentos no ChromaDB local.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    vectordb.persist()
    print("✅ Incidents successfully indexed in vectorstore!")

if __name__ == "__main__":
    os.makedirs("vectorstore", exist_ok=True)
    incident_docs = load_incident_data("data/incidents.csv")
    ingest_documents(incident_docs)
