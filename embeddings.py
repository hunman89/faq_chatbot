import openai

from config import EMBEDDINGS_MODEL, OPENAI_API_KEY


class Embeddings:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def get_embedding(self, texts: list[str], model=EMBEDDINGS_MODEL):
        texts = [text.replace("\n", " ") for text in texts]
        response = openai.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

    def get_batch_embeddings(self, texts: list[str], batch_size=500):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            print(f"{i}/{len(texts)}번째 데이터 임베딩 변환 중...")
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.get_embedding(batch_texts)
            embeddings.extend(batch_embeddings)
        return embeddings
