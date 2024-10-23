import pandas as pd

data = pd.read_pickle('./final_result.pkl')
df = pd.DataFrame(list(data.items()), columns=['question', 'answer'])
print(df.head())