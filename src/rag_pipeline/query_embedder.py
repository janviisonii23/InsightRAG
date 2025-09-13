from sentence_transformers import SentenceTransformer

class QueryEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, query: str):
        return self.model.encode(query).tolist()