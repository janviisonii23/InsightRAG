import os
import uuid
import base64
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.documents.elements import (
    Table, NarrativeText, Title, ListItem, FigureCaption, Image, CodeSnippet
)

from pathlib import Path


class DocumentExtractor:
    def __init__(self, session_id: str, base_dir: str = "../data"):
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / session_id
        self.image_dir = self.session_dir / "images"
        self.table_dir = self.session_dir / "tables"
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

    def _get_elements(self, file):
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            return partition_pdf(
                file=file.file,
                strategy="hi_res",
                extract_image_block_types=["Image", "Table"],
                extract_image_block_output_dir=self.image_dir.as_posix(),
                extract_image_block_to_payload=True
            )
        elif filename.endswith(".docx"):
            return partition_docx(file=file.file)
        else:
            raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")

    def _save_image(self, element):
        image_base64 = getattr(element.metadata, "image_base64", None)
        if image_base64:
            # print(" image base64 found")
            # return None
            img_name = f"img_{uuid.uuid4().hex}.png"
            img_path = self.image_dir / img_name
            img_path = img_path.as_posix()
            with open(img_path, "wb") as img_file:
                img_file.write(base64.b64decode(image_base64))
            return img_path
        
        return None
    
    def _save_table(self, element):
        table_name = f"table_{uuid.uuid4().hex}.html"
        table_path = self.table_dir / table_name
        table_html = getattr(element.metadata, "text_as_html", None)
        table_path = table_path.as_posix()
        with open(table_path, "w", encoding="utf-8") as f:
            if table_html is not None:
                f.write(table_html)
            else:
                f.write(element.text)
        return table_path

    def _finalize_image(self, image_element, caption, context, index):
        image_path = self._save_image(image_element)
        if not image_path:
            return None
        return {
            "type": "image",
            "context": context,
            "path": image_path.replace("../data/", ""),
            "caption": caption,
            "index": index
        }

    def process(self, file):
        elements = self._get_elements(file)
        output = {
            "text_chunks": [],
            "tables": [],
            "images": [],
            "code_snippets": []
        }

        last_text_context = ""
        last_caption = None
        temp_image = None
        index = 0

        for el in elements:
            # Flush any held image before non-caption elements
            if temp_image and not isinstance(el, FigureCaption):
                flushed = self._finalize_image(temp_image, last_caption, last_text_context, index)
                if flushed:
                    output["images"].append(flushed)
                    index += 1
                temp_image = None
                last_caption = None

            if isinstance(el, (NarrativeText, Title, ListItem)):
                text = el.text.strip()
                if text:
                    output["text_chunks"].append({
                        "type": "text",
                        "content": text,
                        "index": index
                    })
                    last_text_context = text
                    index += 1

            elif isinstance(el, Table):
                table_path = self._save_table(el)
                output["tables"].append({
                    "type": "table",
                    "context": last_text_context,
                    "path": table_path.replace("../data/", ""),
                    "html": getattr(el.metadata, "text_as_html", None),
                    "text": el.text,
                    "index": index
                })
                index += 1

            elif isinstance(el, Image):
                temp_image = el  # wait for potential caption

            elif isinstance(el, FigureCaption):
                last_caption = el.text.strip()

            elif isinstance(el, CodeSnippet):
                output["code_snippets"].append({
                    "type": "code_snippet",
                    "context": last_text_context,
                    "text": el.text,
                    "index": index
                })
                index += 1

        # Flush image at end if any left
        if temp_image:
            flushed = self._finalize_image(temp_image, last_caption, last_text_context, index)
            if flushed:
                output["images"].append(flushed)

        return output
