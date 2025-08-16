import streamlit as st 
import pandas as pd 
import pickle
from sklearn.metrics import pairwise_distances_argmin_min

with open('Random_Forest_model.pkl','rb') as file:
    model=pickle.load(file)

with open('scalar_object.pkl','rb') as file:
    obj=pickle.load(file)

df_str = pd.read_csv('df_str.csv')
df_scaled = pd.read_csv('df_scaled.csv')

st.set_page_config(page_title='Car Price Prediction',page_icon='Logo.png')

st.header('Here, we predict car prices, accurately!')

with st.container(border=True):
    col1,col2=st.columns(2)
    make = col1.selectbox("Car make: ",options=df_str['Make'].unique())
    Carmodel = col2.selectbox("Car model: ",options=df_str['Model'].unique())
    year = col1.number_input("Mfg Year of car: ",min_value=2000,max_value=2021)
    engineSize = col2.number_input("Engine Size: ",min_value=1.0,max_value=4.5)
    mileage = col1.number_input("Mileage: ",min_value=50,max_value=200000)
    fuel = col2.selectbox("Fuel Type: ",options=df_str['Fuel Type'].unique())
    transmission = col1.selectbox("Transmission Type: ",df_str['Transmission'].unique())

    make_sc=list(df_str['Make'].unique())
    make_sc.sort()

    car_model_sc = list(df_str['Model'].unique())
    car_model_sc.sort()

    fuel_sc = list(df_str['Fuel Type'].unique())
    fuel_sc.sort()

    transmission_sc = list(df_str['Transmission'].unique())
    transmission_sc.sort()

    input_values=[[make_sc.index(make),car_model_sc.index(Carmodel),year,engineSize,mileage,fuel_sc.index(fuel),transmission_sc.index(transmission)]]
    
    input_values = obj.transform(input_values)

    features = ['Make','Model','Year','Engine Size','Mileage','Fuel Type','Transmission']

    df_new = pd.DataFrame(input_values,columns=features)


    if col2.button("Get Price"):
     df_sc=df_scaled.drop('Unnamed: 0',axis=1)
     out = model.predict(df_new)[0]
     st.write(f"💰 Predicted Price: {out:,.2f}")