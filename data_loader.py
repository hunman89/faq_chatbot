import re
import numpy as np
import pandas as pd
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from config import COLLECTION_NAME
from embeddings import Embeddings


class DataLoader:
    def __init__(self):
        self.db_client = MilvusClient("data/milvus.db")
        self.embeddings = Embeddings()

        if COLLECTION_NAME not in self.db_client.list_collections():

            data = pd.read_pickle('data/final_result.pkl')
            df = pd.DataFrame(list(data.items()), columns=[
                              'question', 'answer'])

            df['question_embedding'] = df['question'].apply(
                self.embeddings.get_embedding)
            df['answer'] = df['answer'].apply(answer_parse)

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
                metric_type="L2",
                index_type="FLAT",
                params={"nlist": 128}
            )
            self.db_client.create_collection(
                "faq_collection", schema=schema, index_params=index_params)

            self.db_client.insert(collection_name="faq_collection",
                                  data=data.to_dict('records'))
            self.db_client.load_collection(collection_name='faq_collection')


def answer_parse(text: str):
    pattern = r"(위 도움말이 도움이 되었나요\?.*?도움말 닫기)"
    text = text.replace("\n", "")
    return re.sub(pattern, '', text, flags=re.DOTALL)
