import requests
import os
from pathlib import Path
from dotenv import load_dotenv

class LLMWrapper:
    def __init__(self, api_key: str = None, model: str = "mistral-7b-instruct"):
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(dotenv_path=env_path)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = f"mistralai/{model}"

        if not self.api_key:
            raise ValueError("OpenRouter API key not provided. Set it in code or as OPENROUTER_API_KEY.")

    def build_prompt(self,context, images, tables, code_snippets):
        prompt = "### Contextual Explanation\n\n"

        if context:
            for doc in context:
                prompt += f"{doc}\n\n"

        if images:
            for path,caption in images:

                prompt += f"![{caption}]({path})\n\n*{caption}*\n\n"

        if tables:
            for tbl in tables:
                prompt += "**Table:**\n" + tbl + "\n\n"

        if code_snippets:
            for code in code_snippets:
                prompt += "```" + code + "\n```\n\n"

        prompt += "### Please explain the above material in a clear, detailed manner, incorporating all elements as appropriate."
        return prompt

    def query(
        self, 
        user_query: str, 
        context: str, 
        image_refs: list, 
        table_refs: list = None, 
        code_snippets: list = None, 
        system_prompt: str = None
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",  # Replace if hosted
        }


        user_prompt = self.build_prompt(context,image_refs,table_refs,code_snippets)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": f"""
You are a helpful assistant. Use only the context provided below to answer the query. Do not hallucinate information. If you don't find enough context, say so.

### Query:
{user_query}
### Context:
{user_prompt}
Please format your answer with headings, bullet points, or code blocks where appropriate.
"""
        })

        payload = {
            "model": self.model,
            "messages": messages,
            # "transform": "middle-out"
        }

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
