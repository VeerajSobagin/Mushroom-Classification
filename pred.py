import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open('save_model/model.pickle', 'rb'))

df = pd.read_csv('/config/workspace/artifact/01042023__150114/data_ingestion/dataset/test.csv')
print(df.head(4))
df= df.drop(columns='class',axis=1)
# print(df.head(2))

label_encoder = LabelEncoder()
for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])
# print(df.head(2))

p = model.predict(df)

df['result']= p
print(df.head(4))