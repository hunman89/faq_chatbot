import openai

from config import OPENAI_API_KEY


class Embeddings:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def get_embedding(self, text: str, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return openai.embeddings.create(input=text, model=model).data[0].embedding
