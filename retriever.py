from pymilvus import MilvusClient
from embeddings import Embeddings


class Retriever:
    def __init__(self):
        self.db_client = MilvusClient("data/milvus.db")
        self.embeddings = Embeddings()

    def retrieve(self, query, threshold=0.8):
        query_vector = self.embeddings.get_embedding(query)

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.db_client.search(
            collection_name="faq_collection",
            data=[query_vector],
            anns_field="question_embedding",
            search_params=search_params,
            limit=1,
            output_fields=["question", "answer"]
        )

        result = results[0][0]
        if result.get('distance') < threshold:
            return {'question': result.get("entity").get("question"), 'answer': result.get("entity").get("answer")}
        return None
