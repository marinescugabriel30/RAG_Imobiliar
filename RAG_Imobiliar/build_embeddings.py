import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Load dataset curat
df = pd.read_csv("properties_clean.csv")

# Functie pentru textul de indexare

def build_index_text(row):
    return (
        f"{row['property_type']} {row['rooms']} camere "
        f"in {row['neighborhood']}, {row['city']}, "
        f"{row['size_sqm']} mp, construit in {row['year_built']}, "
        f"pret {row['price_eur']} euro. "
        f"Etaj {row['floor']} din {row['max_floor']}. "
        f"Parcare: {row['parking']}. "
        f"Distanta metro {row['dist_to_metro_min']} min. "
        f"Descriere: {row['description']}"
    )

df["index_text"] = df.apply(build_index_text, axis=1)

# Incarca modelul de embeddings

print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Creeaza directorul pentru vector store

os.makedirs("vector_store", exist_ok=True)

# Creeaza ChromaDB

print("Creating ChromaDB PersistentClient...")

chroma_client = chromadb.PersistentClient(path="vector_store")

collection = chroma_client.get_or_create_collection(
    name="real_estate_properties",
    metadata={"hnsw:space": "cosine"}
)

# Genereaza embeddings

print("Generating embeddings...")

embeddings = model.encode(
    df["index_text"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

# Adauga embedding-urile in vector store

collection.add(
    ids=df["id"].astype(str).tolist(),
    embeddings=embeddings.tolist(),
    documents=df["index_text"].tolist(),
    metadatas=[
        {
            "property_type": row["property_type"],
            "city": row["city"],
            "neighborhood": row["neighborhood"],
            "price_eur": float(row["price_eur"]),
            "size_sqm": float(row["size_sqm"]),
        }
        for _, row in df.iterrows()
    ]
)

print("Vector store construit cu succes in ./vector_store/")
print("Total proprietati indexate:", len(df))

# Test rapid

query = "apartament 2 camere titan 50 mp"
print("\nTest query:", query)

query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

print("\nPrimele 5 rezultate:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print("----")
    print(doc)
    print(meta)
