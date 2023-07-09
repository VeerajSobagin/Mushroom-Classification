import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import joblib
import base64

st.set_page_config(page_title='Mushrooms Prediction')
st.title('Mushroom Prediction')

upload_file=st.file_uploader('Choose a csv')

def data_clean(df):
    df=df.drop(columns='Unnamed: 0',axis=1)
    encoder = LabelEncoder()
    for column in range(len(df.columns)):
        df[df.columns[column]]= encoder.fit_transform(df[df.columns[column]])
    return df

if upload_file:
    st.markdown('-----')
    data=pd.read_csv(upload_file)
    file=data_clean(df=data)
    loaded_model = joblib.load(open("save_model/model.pkl", 'rb'))
    model=pd.DataFrame(loaded_model.predict(file))
    result=model.replace({0:'Edible' , 1:'Poisons'})
    st.dataframe(result)

download=st.button('Download prediction')
if download:
  'Download Started!'
  csv = result.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()
  linko= f'<a href="data:file/csv;base64,{b64}" download="mushrooms_prediction.csv">Download csv file</a>'
  st.markdown(linko, unsafe_allow_html=True)

  ## python -m streamlit run app.py (to run localy)