import openai

from config import GPT_MODEL, OPENAI_API_KEY


class Generator:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def generate(self, context, query):
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
        ]
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        return response.choices[0].message.content.strip()
