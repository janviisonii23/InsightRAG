import json
from pathlib import Path
from src.extraction.unstructured_extraction import DocumentExtractor
from src.chunking.chunker import StructuredChunker
from src.embedding.chroma_embedder import ChromaEmbedder
from src.rag_pipeline.query_embedder import QueryEmbedder
from src.rag_pipeline.retriever import Retriever
from src.rag_pipeline.context_builder import ContextBuilder
from src.rag_pipeline.llm_wrapper import LLMWrapper

# ---------- CONFIG ----------
file_path = "file_path"  # Your test PDF/DOCX path
session_id = "session_001"     # You can use uuid4 for dynamic session
data_dir = Path("data") / session_id
data_dir.mkdir(parents=True, exist_ok=True)

# ---------- STEP 1: Extract ----------
extractor = DocumentExtractor(session_id=session_id, base_dir="data")
with open(file_path, "rb") as f:
    class FakeUpload:
        filename = Path(file_path).name
        file = f
        content_type = "application/pdf"
    result = extractor.process(FakeUpload())

# Save extracted data to JSON
extracted_path = data_dir / "extracted.json"
with open(extracted_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"✅ Extracted and saved to {extracted_path}")

# ---------- STEP 2: Chunk ----------
chunker = StructuredChunker(input_path=extracted_path)
chunks = chunker.chunk()
chunked_json_path = data_dir/"chunked.json"
with open(chunked_json_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
print(f"✅ Chunked and saved to {chunked_json_path}")

# ---------- STEP 3: Embed ----------
embedder = ChromaEmbedder(chunk_json_path=chunked_json_path,collection_name=session_id)
embedder.embed_and_store()
embedding_path = data_dir/"chroma_db"/session_id
print(f"✅ Embedded and saved to {embedding_path}")
# ---------- STEP 4: Query ----------
user_query = input("\n🔍 Ask your question: ")
query_vector = QueryEmbedder().embed(user_query)

retriever = Retriever(collection_name=session_id)
top_chunks = retriever.retrieve(query_embedding=query_vector, top_k=5)
print(f"✅ Retrieved top {len(top_chunks)} chunks")

# ---------- STEP 5: Context + LLM ----------
builder = ContextBuilder()
context = builder.build(top_chunks)

# system_prompt = """
# You are a helpful assistant that provides clear, well-structured, and well-formatted answers based on the given context.

# Always follow these rules:
# - Use Markdown formatting throughout.
# - Structure the response into numbered major topics and subtopics based on core concepts, not just listing content types.
# - Integrate **all image references** directly in the explanation using `![Alt Text](path)` and provide descriptive captions and contextual relevance.
# - Include all **tables** and **code snippets** exactly as provided, embedding them naturally where they support the explanation.
# - Use bullet points or numbered lists to organize details within subtopics.
# - Use math notation (LaTeX style) where appropriate for equations.
# - Be concise yet thorough, ensuring no referenced visual or code element is omitted.
# - Write in a conversational, academic style suitable for graduate-level readers.

# Never treat images, tables, or code snippets as separate sections; always embed them smoothly into the flow of your explanation.
# """
system_prompt = """
You are an intelligent documentation assistant built to help users understand complex documents like financial guides, software manuals, and scientific papers.

You must always follow these formatting and response rules:

1. **Markdown Output**
   - Your final answer must be fully written in **Markdown syntax**.
   - Include all visual elements (images, tables, code) inline with proper formatting.
   - Start every major section with a Markdown heading (##, ###) based on context hierarchy.

2. **Images**
   - If any image path is provided in the context, embed it directly using:
     `![Descriptive Alt Text](data/session_<session_id>/images/<filename>.png)`
   - The image must be accompanied by a **clear, helpful caption** explaining its relevance.

3. **Tables**
   - If table HTML or text is provided, insert it using Markdown's triple backticks:
     ```html
     <!-- Table -->
     <table>...</table>
     ```
   - Mention the table title or summary **before** the table.

4. **Code Snippets**
   - Wrap all code blocks in triple backticks with correct language syntax:
     ```python
     def example():
         print("Hello World")
     ```

5. **Structure**
   - Organize your response into **numbered major sections** and **bullet-point sublists**.
   - Maintain academic tone: concise, graduate-level, and easy to follow.
   - If math equations are needed, use LaTeX-style Markdown (e.g., `$E = mc^2$`).

6. **Accuracy and Relevance**
   - Only use provided context to answer.
   - Avoid hallucination. Do not invent image filenames or paths.
   - Embed only the visual/code/table elements actually present in the context.

7. **Consistency**
   - Always use the same path format: `data/session_<session_id>/images/<filename>.png`
   - Ensure every chunk of content has its accompanying image/table/code in the **same section**.

Never output plain text explanations without Markdown formatting. Never place visuals at the end or outside of the explanation.
"""

llm = LLMWrapper()
response = llm.query(
    user_query=user_query,
    context=context["context"],
    image_refs=context["images"],
    table_refs=context["tables"],
    code_snippets=context["code"],
    system_prompt=system_prompt
)

# ---------- OUTPUT ----------
print("\n📘 Final Answer:\n")
print(response)
