import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

st.title('Predicting used cars!')
st.image('./cars.jpg')
st.write('This dataset is the stacked version of [100,000 UK Used Car Data](https://www.kaggle.com/aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw) present in Kaggle. Here we have combined the used car information of 7 brands namely Audi, BMW, Skoda, Ford, Volkswagen, Toyota and Hyundai.')
st.header('Features for predicting the car prices')
st.markdown('**model:** Model for the car')
st.markdown('**year:** When the car was manufactured')
st.markdown('**transmission:** Mode for the car')
st.markdown('**mileage:** Number of miles the car has been driven')
st.markdown('**fuelType:** The type of fuel the car uses')
st.markdown('**Tax:** Road Tax for the car')
st.markdown('**mpg:** How many miles your car can go per gallon')
st.markdown('**engineSize:** Engine size is the volume of fuel and air that can be pushed through a cars cylinders, unit in (cc)')
st.markdown('**Make:** Name of the car company')

st.header('Target')
st.markdown('**Price:** Price of the car we are trying to predict')
st.markdown('## Preview of the dataframe')
df = pd.read_csv('cars_dataset.csv')
st.write(df.head())

#Cleaning the Dataframe.
df.drop(df[df['mpg']<15].index,inplace=True)
df.drop(df[df['engineSize']<1].index,inplace=True)
df['fuelType'] = df['fuelType'].replace('Electric', 'Other')
df['transmission'] = df['transmission'].replace('Other', 'Manual')

#Subsetting Model for corresponding Manufacturer.
make_model_map = {}

for make in df.Make.unique():
  model_list = df[df['Make']==make]['model'].unique()
  make_model_map[make] = list(model_list)



st.header('Start selecting the features of your car below')
st.subheader('Pick the year when the car was made')

#Selecting feature values and storing them in variables.
year = st.selectbox('',sorted(df.year.unique()))

st.subheader('Pick the car company from the list')
make = st.selectbox('',sorted(df.Make.unique()))

st.subheader('Pick the model for the car')
model = st.selectbox('',make_model_map[make])

st.subheader('Pick the car mode')
transmission = st.selectbox('',df.transmission.unique())

st.subheader('Pick the kind of fuel')
fuelType = st.radio('',df.fuelType.unique())
st.subheader('Slide to pick the mileage of the car')
mileage = st.slider('',min_value=df.mileage.min(), max_value=df.mileage.max(), value=150000)
st.subheader('Slide to choose the mpg')
mpg = st.slider('', min_value=df.mpg.min(), max_value=df.mpg.max(), value=250.0)
st.subheader('Slide to choose the road tax')
tax = st.slider('', min_value=df.tax.min(), max_value=df.tax.max(), value=180.0)
st.subheader('Pick the engineSize')
engineSize= st.slider('',min_value=df.engineSize.min(), max_value=df.engineSize.max(), value=3)

#Instantiating and appending new data for prediction.
inference_data =[]
inference_data.append([model,year,transmission,mileage,fuelType,tax,mpg,engineSize,make])
inference_data_df = pd.DataFrame(inference_data,columns=['model','year','transmission','mileage','fuelType','tax','mpg','engineSize','Make'])

#Label Encoding String values.
cat_cols = list(df.select_dtypes('object').columns)
num_cols = [cols for cols in df.columns if cols not in cat_cols]
le =LabelEncoder()
for c in cat_cols:
  inference_data_df[c] = le.fit_transform(inference_data_df[c])

#Model Selection
st.subheader('Pick an algorithm')
selected_model = st.selectbox('',[ 'Decision Tree Regressor','Linear Regression', 'ElasticNet', 'Ridge'])

#Function for model prediction.
def model_switcher(model):
    if model == 'Decision Tree Regressor':
        loaded_model = joblib.load('models/dtr_model.pkl')
        result = loaded_model.predict(inference_data_df[0:1])
        return '£' + str(round(abs(result[0]),2)) 

    elif model == 'Linear Regression':
        loaded_model = joblib.load('models/lr_model.pkl')
        result = loaded_model.predict(inference_data_df[0:1])
        return '£' + str(round(abs(result[0]),2))

    elif model == 'ElasticNet':
        loaded_model = joblib.load('models/Elastic_model.pkl')
        result = loaded_model.predict(inference_data_df[0:1])
        return '£' + str(round(abs(result[0]),2))

    elif model == 'Ridge':
        loaded_model = joblib.load('models/Ridge_model.pkl')
        result = loaded_model.predict(inference_data_df[0:1])
        return '£' + str(round(abs(result[0]),2))      
    else:
        return None

st.header('Prediction!')
st.write('Your car could be worth....',model_switcher(selected_model))
