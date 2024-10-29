import re
import pandas as pd
from pymilvus import CollectionSchema, DataType, FieldSchema

from config import COLLECTION_NAME
from embeddings import Embeddings


class DataLoader:
    def __init__(self, db_client):
        self.db_client = db_client
        self.embeddings = Embeddings()
        self.initialize_collection()

    def initialize_collection(self):
        if COLLECTION_NAME not in self.db_client.list_collections():
            data = pd.read_pickle("data/final_result.pkl")
            df = pd.DataFrame(list(data.items()), columns=[
                            'question', 'answer'])

            df['question_embedding'] = self.embeddings.get_batch_embeddings(
                df['question'].tolist())
            df['answer'] = df['answer'].apply(self.answer_parse)

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64,
                            is_primary=True, auto_id=True),
                FieldSchema(name="question_embedding", dtype=DataType.FLOAT_VECTOR,
                            dim=len(df['question_embedding'][0])),
                FieldSchema(name="question",
                            dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="answer", dtype=DataType.VARCHAR,
                            max_length=20000)
            ]
            schema = CollectionSchema(fields, description="FAQ chatbot data")
            index_params = self.db_client.prepare_index_params()
            index_params.add_index(
                field_name="question_embedding",
                metric_type="COSINE",
                index_type="FLAT"
            )
            self.db_client.create_collection(
                "faq_collection", schema=schema, index_params=index_params)

            self.db_client.insert(collection_name="faq_collection",
                                data=df.to_dict('records'))
            self.db_client.load_collection(collection_name='faq_collection')

    @staticmethod
    def answer_parse(text: str):
        pattern = r"(위 도움말이 도움이 되었나요\?.*?도움말 닫기)"
        text = text.replace("\n", "")
        return re.sub(pattern, '', text, flags=re.DOTALL)
