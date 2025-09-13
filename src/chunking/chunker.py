import json
from pathlib import Path

class StructuredChunker:
    def __init__(self, input_path, max_words=300):
        self.input_path = Path(input_path)
        self.max_words = max_words

    def load_data(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def chunk(self):
        doc = self.load_data()
        all_items = []

        # Combine all items into a single list sorted by `index`
        for key in ["text_chunks", "tables", "images", "code_snippets"]:
            for item in doc.get(key, []):
                all_items.append(item)

        all_items.sort(key=lambda x: x.get("index", 0))

        chunks = []
        current_chunk = {
            "content": [],
            "images": [],
            "tables": [],
            "code_snippets": [],
            "word_count": 0
        }

        for item in all_items:
            text = item.get("content") or item.get("text") or ""
            num_words = len(text.strip().split())

            # Flush current chunk if it exceeds max words
            if current_chunk["word_count"] + num_words > self.max_words and current_chunk["content"]:
                chunks.append({
                    "content": " ".join(current_chunk["content"]),
                    "images": current_chunk["images"],
                    "tables": current_chunk["tables"],
                    "code_snippets": current_chunk["code_snippets"]
                })
                current_chunk = {
                    "content": [],
                    "images": [],
                    "tables": [],
                    "code_snippets": [],
                    "word_count": 0
                }

            # Add item based on type
            if item["type"] == "text":
                current_chunk["content"].append(text)
                current_chunk["word_count"] += num_words

            elif item["type"] == "table":
                current_chunk["tables"].append({
                    "path": item.get("path"),
                    "html": item.get("html"),
                    "text": item.get("text"),
                    "context": item.get("context")
                })

            elif item["type"] == "image":
                current_chunk["images"].append({
                    "path": item.get("path"),
                    "caption": item.get("caption"),
                    "context": item.get("context")
                })

            elif item["type"] == "code_snippet":
                current_chunk["code_snippets"].append({
                    "text": item.get("text"),
                    "context": item.get("context")
                })

        # Flush last chunk
        if current_chunk["content"]:
            chunks.append({
                "content": " ".join(current_chunk["content"]),
                "images": current_chunk["images"],
                "tables": current_chunk["tables"],
                "code_snippets": current_chunk["code_snippets"]
            })

        return chunks
