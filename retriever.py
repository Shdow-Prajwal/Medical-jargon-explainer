import chromadb, ollama

client = chromadb.PersistentClient(path="./db")
col = client.get_or_create_collection("pages", metadata={"hnsw:space": "cosine"})

def _embed(texts, kind):  # kind: "search_document" or "search_query"
    r = ollama.embed(model="nomic-embed-text",
                     input=[f"{kind}: {t[:6000]}" for t in texts])
    return r["embeddings"]

def index_page(doc_id, page_no, text, image_path):
    col.upsert(
        ids=[f"{doc_id}:{page_no}"],
        documents=[text],
        embeddings=_embed([text], "search_document"),
        metadatas=[{"doc_id": doc_id, "page": page_no, "image_path": image_path}],
    )

def search(query, doc_id, k=3):
    r = col.query(query_embeddings=_embed([query], "search_query"),
                  n_results=k, where={"doc_id": doc_id})
    return [{**m, "distance": d} for m, d in zip(r["metadatas"][0], r["distances"][0])]