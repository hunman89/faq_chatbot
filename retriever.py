import openai
from pymilvus import MilvusClient

from config import OPENAI_API_KEY
from embeddings import Embeddings


class Retriever:
    def __init__(self):
        self.db_client = MilvusClient("data/milvus.db")
        self.embeddings = Embeddings()

    def retrieve(self, query, top_k=5):
        query_vector = self.embeddings.get_embedding(query)

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.db_client.search(
            collection_name="faq_collection",
            data=[query_vector],
            anns_field="question_embedding",
            search_params=search_params,
            limit=top_k,
            output_fields=["question", "answer"]
        )

        documents = [hit.get("entity").get("answer") for hit in results[0]]
        return documents
