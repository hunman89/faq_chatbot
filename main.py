import pandas as pd
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

data = pd.read_pickle('./final_result.pkl')
df = pd.DataFrame(list(data.items()), columns=['question', 'answer'])

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

df['question_embedding'] = df['question'].apply(get_embedding)
df.to_csv('embedded.csv')