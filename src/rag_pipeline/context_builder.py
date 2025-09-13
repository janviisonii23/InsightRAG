import json

class ContextBuilder:
    def build(self, results):
        # print(results)

        chunks = results["documents"][0]
        metadata_list = results["metadatas"][0]

        context = "\n\n".join(chunks)

        image_refs = []
        table_refs = []
        code_snippets = []

        for meta in metadata_list:

            images_json_str = meta.get("images","[]")
            try:
                images = json.loads(images_json_str)
                image_refs.extend([(img["path"], img.get("caption","No Caption")) for img in images])
            except Exception as e:
                print(f"Error parsing images: {e}")

            table_refs.extend(meta.get("tables", []))
            code_snippets.extend(meta.get("code", []))

        return {
            "context": context,
            "images": image_refs,
            "tables": table_refs,
            "code": code_snippets
        }
