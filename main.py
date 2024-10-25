import re
import numpy as np
import pandas as pd
from openai import OpenAI
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient
from config import OPENAI_API_KEY

# pattern = r"(위 도움말이 도움이 되었나요\?.*?도움말 닫기)"
client = OpenAI(api_key=OPENAI_API_KEY)

# data = pd.read_pickle('data/final_result.pkl')
# df = pd.DataFrame(list(data.items()), columns=['question', 'answer'])


def get_embedding(text: str, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# def answer_parse(text: str):
#     text = text.replace("\n", "")
#     return re.sub(pattern, '', text, flags=re.DOTALL)


# df['question_embedding'] = df['question'].apply(get_embedding)
# df['answer'] = df['answer'].apply(answer_parse)

# df.to_csv('data/data.csv', index=False)

# data = pd.read_csv('data/data.csv',
#                    dtype={'question': str, 'answer': str, 'question_embedding': object})
# data['question_embedding'] = data['question_embedding'].apply(
#     eval).apply(np.array)

db_client = MilvusClient("data/milvus.db")

# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64,
#                 is_primary=True, auto_id=True),
#     FieldSchema(name="question_embedding", dtype=DataType.FLOAT_VECTOR,
#                 dim=len(data['question_embedding'][0])),
#     FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=100),
#     FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=10000)
# ]
# schema = CollectionSchema(fields, description="FAQ chatbot data")

# if client.has_collection("faq_collection"):
#     client.drop_collection("faq_collection")
# client.create_collection("faq_collection", schema=schema)

# res = client.insert(collection_name="faq_collection",
#                     data=data.to_dict('records'))
# print(res)

# index_params = db_client.prepare_index_params()
# index_params.add_index(
#     field_name="question_embedding",
#     metric_type="L2",
#     index_type="FLAT",
#     params={"nlist": 128}
# )
# db_client.create_index(collection_name='faq_collection',
#                        index_params=index_params)
# db_client.load_collection(collection_name='faq_collection')

query = "스마트스토어 회원가입은 어떻게 하나요?"
query_vector = get_embedding(query)

search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = db_client.search(
    collection_name="faq_collection",
    data=[query_vector],
    anns_field="question_embedding",
    search_params=search_params,
    limit=5,
    output_fields=["question", "answer"]
)

documents = [hit.get("entity").get("question") for hit in results[0]]
print(documents)
