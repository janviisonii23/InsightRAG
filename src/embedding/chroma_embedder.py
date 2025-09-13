import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb


class ChromaEmbedder:
    def __init__(self, chunk_json_path, persist_dir="../data/chroma_db", collection_name="session_id"):
        self.chunk_json_path = Path(chunk_json_path)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def load_chunks(self):
        with open(self.chunk_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def embed_and_store(self):
        chunks = self.load_chunks()

        for idx, chunk in enumerate(chunks):
            text = chunk.get("content", "").strip()
            if not text:
                continue  # skip empty text chunks

            metadata = {
                "images": json.dumps(chunk.get("images", [])),
                "tables": json.dumps(chunk.get("tables", [])),
                "code_snippets": json.dumps(chunk.get("code_snippets", [])),
                "source_id": str(idx)
            }

            embedding = self.embedder.encode(text).tolist()

            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[str(idx)],
                metadatas=[metadata]
            )

        # print(f"{len(chunks)} chunks embedded and stored in ChromaDB at '{self.persist_dir}'")
