import re
import numpy as np
import pandas as pd
from openai import OpenAI
from pymilvus import MilvusClient
from config import OPENAI_API_KEY

pattern = r"(위 도움말이 도움이 되었나요\?.*?도움말 닫기)"
client = OpenAI(api_key=OPENAI_API_KEY)

data = pd.read_pickle('data/final_result.pkl')
df = pd.DataFrame(list(data.items()), columns=['question', 'answer'])


def get_embedding(text: str, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def answer_parse(text: str):
    text = text.replace("\n", "")
    return re.sub(pattern, '', text, flags=re.DOTALL)


df['question_embedding'] = df['question'].apply(get_embedding)
df['answer'] = df['answer'].apply(answer_parse)

df.to_csv('data/data.csv', index=False)
