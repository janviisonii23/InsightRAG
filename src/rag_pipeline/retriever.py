import chromadb

class Retriever:
    def __init__(self, db_path="../data/chroma_db", collection_name="session_id"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def retrieve(self, query_embedding, top_k=5):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results