"""
Build a ChromaDB index of BNS sections using BAAI/bge-m3.

- Embedded text : "<Section name>. <Chapter name>. <Description>"
- Metadata      : every CSV column + case_type = "criminal cases"
- One record per section (BGE-M3 handles up to 8192 tokens, so no chunking needed)

Install:  pip install chromadb FlagEmbedding pandas
Use:      run this file / cell, then call build_index() once and search("...") as needed.
"""
import threading
from pathlib import Path

import chromadb
import pandas as pd

# ---------- settings ----------
# Paths are anchored to the project root (the folder above src/), so it does not matter
# where you launch from - the same bns_chroma folder is always used.
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = str(BASE_DIR / "Data" / "bns_sections.csv")
DB_PATH = str(BASE_DIR / "bns_chroma")
COLLECTION = "bns_sections"
MODEL_NAME = "BAAI/bge-m3"
CASE_TYPE = "criminal cases"

_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        from FlagEmbedding import BGEM3FlagModel

        # fp16 speeds things up on GPU; keep it off on CPU
        _model = BGEM3FlagModel(MODEL_NAME, use_fp16=torch.cuda.is_available())
    return _model


def embed(texts, batch_size=8):
    """Dense embeddings from BGE-M3 (already normalised; no query/passage prefix needed)."""
    out = get_model().encode(texts, batch_size=batch_size, max_length=8192)["dense_vecs"]
    return out.astype("float32").tolist()


def load_records(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df.columns = [c.replace(" ", "") for c in df.columns]  # 'Section _name' -> 'Section_name'
    for c in ["Chapter_name", "Chapter_subtype", "Section_name", "Description"]:
        df[c] = df[c].astype(str).str.strip()

    ids, docs, metas = [], [], []
    for r in df.itertuples(index=False):
        ids.append(f"bns_s{r.Section}")
        docs.append(f"{r.Section_name}. {r.Chapter_name}. {r.Description}")
        metas.append(
            {
                "case_type": CASE_TYPE,
                "source": "Bharatiya Nyaya Sanhita, 2023",
                "Chapter": int(r.Chapter),
                "Chapter_name": r.Chapter_name,
                "Chapter_subtype": r.Chapter_subtype,
                "Section": int(r.Section),
                "Section_name": r.Section_name,
            }
        )
    return ids, docs, metas


def build_index(csv_path=CSV_PATH, db_path=DB_PATH):
    ids, docs, metas = load_records(csv_path)
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine", "embedder": MODEL_NAME}
    )
    embeddings = embed(docs)
    for i in range(0, len(ids), 128):
        col.add(
            ids=ids[i : i + 128],
            documents=docs[i : i + 128],
            embeddings=embeddings[i : i + 128],
            metadatas=metas[i : i + 128],
        )
    print(f"Indexed {len(ids)} sections into {db_path}")
    return col


_BUILD_LOCK = threading.Lock()


def _open_if_ready(db_path):
    """Return the collection if it exists and has data, else None."""
    try:
        col = chromadb.PersistentClient(path=db_path).get_collection(COLLECTION)
        return col if col.count() > 0 else None
    except Exception:  # collection (or the DB) does not exist yet
        return None


def get_collection(db_path=DB_PATH):
    """Open the BNS index; if it is missing or empty, build it first (first run only, takes a few minutes)."""
    col = _open_if_ready(db_path)
    if col is not None:
        return col
    with _BUILD_LOCK:  # several Streamlit sessions must not build at the same time
        col = _open_if_ready(db_path)  # someone may have finished while we waited
        if col is None:
            print(f"BNS index not found in {db_path} - building it now (first run only)...")
            col = build_index(db_path=db_path)
    return col


def search(query, k=5, db_path=DB_PATH):
    """Semantic search, restricted to criminal-case records."""
    res = get_collection(db_path).query(
        query_embeddings=embed([query]),
        n_results=k,
        where={"case_type": CASE_TYPE},
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"section": meta["Section"], "name": meta["Section_name"],
                     "chapter": meta["Chapter_name"], "distance": dist,
                     "metadata": meta, "text": doc})
    return hits


def get_section(section_number, db_path=DB_PATH):
    """Exact lookup by section number (metadata, not vector). Returns None if it doesn't exist."""
    res = get_collection(db_path).get(where={"Section": int(section_number)})
    if not res["ids"]:
        return None
    return {"metadata": res["metadatas"][0], "text": res["documents"][0]}


if __name__ == "__main__":
    build_index()
    for h in search("someone threatened to kill me"):
        print(h["section"], h["name"], round(h["distance"], 3))