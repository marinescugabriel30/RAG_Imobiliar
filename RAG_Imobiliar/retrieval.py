import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import re
import json

# LOAD MODEL + VECTOR STORE

model = SentenceTransformer("all-MiniLM-L6-v2")

# Initializare Chroma
chroma_client = chromadb.PersistentClient(path="vector_store")

# Preia colectia
collection = chroma_client.get_or_create_collection(
    name="real_estate_properties",
    metadata={"hnsw:space": "cosine"}
)

df = pd.read_csv("properties_clean.csv")
df = df.set_index("id")

# FILTER EXTRACTION

def extract_filters(user_query: str):
    q = user_query.lower()
    filters = {}

    # Property type
    if "apartament" in q:
        filters["property_type"] = "apartment"
    elif "casa" in q or "vila" in q or "vilă" in q:
        filters["property_type"] = "house"
    elif "teren" in q:
        filters["property_type"] = "land"

    # Rooms
    m = re.search(r"(\d+)\s*cam", q)
    if m:
        filters["rooms"] = int(m.group(1))

    # Price max
    m = re.search(r"(\d{2,6})\s*euro", q)
    if m:
        filters["price_max"] = int(m.group(1))

    # Neighborhood detection
    neighborhoods = ["titan", "militari", "dristor", "berceni", "aviatiei", "pipera", "drumul taberei"]
    for nb in neighborhoods:
        if nb in q:
            filters["neighborhood"] = nb.title()

    return filters

# RERANKING FUNCTION

def rerank_results(raw_results, query_embedding, filters):
    reranked = []

    for item in raw_results:
        prop_id = int(item["id"])
        row = df.loc[prop_id]

        doc_emb = np.array(item["embedding"])
        q_emb = np.array(query_embedding)

        # Similaritate cosinus
        similarity = float(
            np.dot(q_emb, doc_emb) /
            (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb))
        )

        score = similarity

        # Bonus pentru cartier identic
        if "neighborhood" in filters:
            if row["neighborhood"].lower() == filters["neighborhood"].lower():
                score += 0.20
            else:
                score -= 0.10

        # Penalizare pentru pret/mp diferit de "ideal"
        if row["price_per_sqm"] > 0:
            ideal_ppsqm = None

            if "price_max" in filters and row["size_sqm"] > 0:
                ideal_ppsqm = filters["price_max"] / row["size_sqm"]

            if ideal_ppsqm:
                diff = abs(row["price_per_sqm"] - ideal_ppsqm)
                score -= min(diff / 5000, 0.15)

        reranked.append({
            "id": int(prop_id),
            "similarity": float(similarity),
            "final_score": float(score),
            "property_type": str(row["property_type"]),
            "neighborhood": str(row["neighborhood"]),
            "city": str(row["city"]),
            "price_eur": int(row["price_eur"]),
            "size_sqm": int(row["size_sqm"]),
            "price_per_sqm": float(row["price_per_sqm"]),
        })

    # Sortare descrescatoare
    reranked = sorted(reranked, key=lambda x: x["final_score"], reverse=True)
    return reranked

# MAIN RETRIEVAL FUNCTION

def get_comparables(user_query: str, k=10):
    print("\nUser query:", user_query)

    filters = extract_filters(user_query)
    print("Extracted filters:", filters)

    q_emb = model.encode([user_query])[0]

    # Query vector store
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=50,
        include=["embeddings"]    # necesar pentru reranking
    )

    # Construim lista
    raw_items = []
    for i in range(len(results["ids"][0])):
        raw_items.append({
            "id": int(results["ids"][0][i]),
            "embedding": results["embeddings"][0][i],
        })

    # Reranking
    ranked = rerank_results(raw_items, q_emb, filters)
    topk = ranked[:k]

    # Save output
    with open("comparables.json", "w", encoding="utf-8") as f:
        json.dump(topk, f, indent=4, ensure_ascii=False)

    print("Saved: comparables.json")
    return topk

# TEST

if __name__ == "__main__":
    res = get_comparables("apartament 2 camere titan buget 60000 euro")
    print("\nTOP comparables:")
    for x in res:
        print(x, "\n")
