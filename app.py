
# coding: utf-8

# In[17]:


import os
import uuid
import sys
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import *
from sklearn import preprocessing
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from azure.storage.blob import BlockBlobService,PublicAccess
from functools import cmp_to_key


# In[14]:


#Title of the web page
st.title("Predicting next day rain in Australia")

def label_encoding(df):
    #Dropping columns with NULL values
    df = df.dropna(axis=0)
    #Dropping NULL values
    df = df.dropna()
    #Label encoding
    le = LabelEncoder()
    df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
    df['WindDir9am'] = le.fit_transform(df['WindDir9am'])
    df['WindDir3pm'] = le.fit_transform(df['WindDir3pm'])
    df['RainToday'] = le.fit_transform(df['RainToday'])
    return df


# In[15]:


#Function to create a table for CSV data input with many rows
def table(rainTomorrow,location,date):
    df_final = pd.DataFrame(list(zip(date,location,rainTomorrow)),columns = ['Date','Location','RainTomorrow'])
    st.table(df_final)


# In[23]:



def download_pickle():
    block_blob_service = BlockBlobService(account_name='manavblob',account_key='+st/VdshIg7EcBNZ7HrUHp2AauMlwuIIhufkaXtwKqcNxbHsycjrMac45cCtSAKQ6CtiLfz/E7G3DGnrLVUHHQ==')
    block_blob_service.set_container_acl('model',public_access=PublicAccess.Container)
    generator = block_blob_service.list_blobs('model')
    blob_list=[]
    for blob in generator:
        blob_list.append((blob.name))
    blob_list.sort(reverse=True)
    print(blob_list)
    block_blob_service.get_blob_to_path('model',blob_list[0],'C:\\Users\\manav\\cloud_downloads\\model1.pkl')
    f = open('C:\\Users\\manav\\cloud_downloads\\model1.pkl', 'rb')
    classifier = pickle.load(f)
    return classifier


# In[17]:


#Function to generate predictions for CSV data input
def op(df,location,date):
    classifier = download_pickle()
    predictions = classifier.predict(df)
    rainTomorrow = []
    for i in predictions:
        if(predictions[i]==1):
            rainTomorrow.append('Yes')
        else:
            rainTomorrow.append('No')
    table(rainTomorrow,location,date)
    


# In[18]:


#Function to generate predictions for single row of data      
def op1(df,location,date):
    classifier = download_pickle()
    predictions = classifier.predict(df)
    if(predictions[0]==1):
            rainTomorrow = 'Yes'
    else:
            rainTomorrow = 'No'
    print(rainTomorrow,location,date)
    df_final = pd.DataFrame(columns = ['Date','Location','RainTomorrow'])
    values = [date,location,rainTomorrow]
    df_final.loc[len((df_final.index))+1] = values
    st.table(df_final)


# In[19]:


#Function to allow the user to input one row of data
def user_io():
        date = st.date_input("Date")
        st.write('You selected:', date)
        location = st.selectbox("Location",("Adelaide","Albany","Albury","AliceSprings","BadgerysCreek","Ballarat","Bendigo","Brisbane","Cairns","Canberra","Cobar","CoffsHarbour","Dartmoor","Darwin","GoldCoast","Hobart","Katherine","Launceston","Melbourne","MelbourneAirport","Mildura","Moree","MountGambier","MountGinini","Newcastle","Nhil","NorahHead","NorfolkIsland","Nuriootpa","PearceRAAF","Penrith","Perth","PerthAirport","Portland","Richmond","Sale","SalmonGums","Sydney","SydneyAirport","Townsville","Tuggeranong","Uluru","WaggaWagga","Walpole","Watsonia","Williamtown","Witchcliffe","Wollongong","Woomera"))
        st.write('You selected:', location)
        minTemp = st.number_input("Minimum Temperature")
        maxTemp = st.number_input("Maximum Temperature")
        rainfall = st.number_input("Rainfall")
        windGustDirection = st.selectbox("WindGustDirection",("E","ENE","ESE","N","NA","NE","NNE","NNW","NW","S","SE","SSE","SSW","SW","W","WNW","WSW"))
        st.write('You selected:', windGustDirection)
        windGustSpeed = st.number_input("Wind Gust Speed")
        windDirection9AM = st.selectbox("Wind Direction at 9AM",("E","ENE","ESE","N","NA","NE","NNE","NNW","NW","S","SE","SSE","SSW","SW","W","WNW","WSW"))
        st.write('You selected:', windDirection9AM)
        windDirection3PM = st.selectbox("Wind Direction at 3PM",("E","ENE","ESE","N","NA","NE","NNE","NNW","NW","S","SE","SSE","SSW","SW","W","WNW","WSW"))
        st.write('You selected:', windDirection3PM)
        windSpeed9AM = st.number_input("Wind Speed at 9AM")
        windSpeed3PM = st.number_input("Wind Speed at 3PM")
        humidity9AM = st.number_input("Humidity at 9AM")
        humidity3PM = st.number_input("Humidity at 3PM")
        pressure9AM = st.number_input("Pressure at 9AM")
        pressure3PM = st.number_input("Pressure at 3PM")
        cloud9AM = st.number_input("Cloud at 9AM")
        cloud3PM = st.number_input("Cloud at 3PM")
        temp9AM = st.number_input("Temperature at 9AM")
        temp3PM = st.number_input("Temperature at 3PM")
        rainToday = st.selectbox("RainToday",(0,1))
        st.write('You selected:', rainToday)
        button = st.button("Submit")
        if(button == True):
            user_data = {
                   'Date':date,
                    'Location':location,
                    'MinTemp':minTemp,
                    'MaxTemp':maxTemp,
                    'Rainfall':rainfall,
                    'WindGustDir':windGustDirection,
                    'WindGustSpeed':windGustSpeed,
                    'WindDir9am':windDirection9AM,
                    'WindDir3pm':windDirection3PM,
                    'WindSpeed9am':windSpeed9AM,
                    'WindSpeed3pm':windSpeed3PM,
                    'Humidity9am':humidity9AM,
                    'Humidity3pm':humidity3PM,
                    'Pressure9am':pressure9AM,
                    'Pressure3pm':pressure3PM,
                    'Cloud9am':cloud9AM,
                    'Cloud3pm':cloud3PM,
                    'Temp9am':temp9AM,
                    'Temp3pm':temp3PM,
                   'RainToday':rainToday,
               }
            row = pd.DataFrame(user_data,index=[0])
            location = row.iloc[0]['Location']
            date = row.iloc[0]['Date']
            row = row.drop(["Cloud9am","Cloud3pm","Location","Date"],axis=1)
            new_row = label_encoding(row)
            op1(new_row,location,date) 
        else:
            st.text("Please click on the Submit button")


# In[20]:


#Main function
def main():  
    choice = st.selectbox("How do you want to input your data?",("Upload a CSV file","Enter datafields manually"))
    if(choice=="Upload a CSV file"):
        csv = st.file_uploader("Upload a CSV file",type =["csv"])
        button =  st.button("Submit")
        if(button==True):
            df = pd.read_csv(csv)
            location = []
            date = []
            n = df.shape[0]
            for r in range(n):
                location.append(df.iloc[r]['Location'])
                date.append(df.iloc[r]['Date'])
            print(location)
            print(date)
            df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Date','Location','RainTomorrow'],axis=1)
            print(df)
            new_df = label_encoding(df)
            op(new_df,location,date)
        else:
            st.text("Please click on the Submit button")
    else:
        user_io() 
#Calling Main function
if __name__ == "__main__":
    main()

