#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# In[2]:


# importer la data :
data= pd.read_csv(r"C:\Users\admin\Desktop\Nouveau dossier (7)\application_train.csv")
data= data.dropna()
data=data[2:100]


# In[4]:


# model= pickle.load(open("model_pkl","rb"))
model= pickle.load(open("model_pkl","rb"))


# In[5]:


st.title("**♟**Tableau de bord**♟**")


# In[6]:


CODE_GENDER = st.sidebar.multiselect("CODE_GENDER", data['CODE_GENDER'].unique())
if CODE_GENDER:
    data = data[data.CODE_GENDER.isin(CODE_GENDER)]


# In[ ]:


labels = '0', '1'
sizes = [data['TARGET'].value_counts()[0],data['TARGET'].value_counts()[1]]
fig1, ax1 = plt.subplots(figsize=[6,6])
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("               Répartition des clients selon leurs probabilité de faire défaut dans le paiement de leurs crédit", fontsize=14)
#st.plotly_chart (fig1)
st.pyplot(fig1)





colonne= ["EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1", "AMT_CREDIT","AMT_INCOME_TOTAL", "REGION_POPULATION_RELATIVE","AMT_GOODS_PRICE","AMT_ANNUITY","DAYS_EMPLOYED","SK_ID_CURR","DAYS_REGISTRATION","DAYS_LAST_PHONE_CHANGE","HOUR_APPR_PROCESS_START","DAYS_ID_PUBLISH","DAYS_BIRTH"]

# # Regrouper:




import seaborn as sns
sns.set(rc={"figure.figsize":(3, 5)}) 
fig1, ax1 = plt.subplots()
ax1 = sns.boxplot(y=data["EXT_SOURCE_1"],palette= "autumn_r")
plt.title("Boite à moustache de la variable Ext_source 1", fontsize=14)

sns.set(rc={"figure.figsize":(3, 5)}) 
fig2, ax2 = plt.subplots()
ax2 = sns.boxplot(y=data["EXT_SOURCE_2"],palette= "autumn_r")
plt.title("Boite à moustache de la variable Ext_source 2", fontsize=14)


sns.set(rc={"figure.figsize":(3, 5)}) 
fig3, ax3 = plt.subplots()
ax3 = sns.boxplot(y=data["EXT_SOURCE_3"],palette= "autumn_r")
plt.title("Boite à moustache de la variable Ext_source 3", fontsize=14)

SK_ID_CURR= st.sidebar.selectbox("Client ID",data['SK_ID_CURR'].unique() )

if SK_ID_CURR:
    prediction= model.predict_proba(np.array(data.loc[data['SK_ID_CURR']== SK_ID_CURR,colonne]).astype(np.float64))
    st.success('La probabilité que le client fait défaut dans le paiement de son crédit est {}'.format(prediction[0][0]))
    if prediction[0][0] > 0.5:
            st.success(" Donc l'attribution de ce crédit est risqué")
    else:
            st.success("Donc l'attribution de ce crédit est sûre")
    st.subheader("Ci-dessous vous trouverez les caractéristiques les plus influençants sur ce score et la position du client dans cette variable par rapport aux autres clients :")


    ax1.axhline(float(data.loc[data['SK_ID_CURR']== SK_ID_CURR,'EXT_SOURCE_1']),ls='--' )
    ax2.axhline(float(data.loc[data['SK_ID_CURR']== SK_ID_CURR,'EXT_SOURCE_2']),ls='--' )
    ax3.axhline(float(data.loc[data['SK_ID_CURR']== SK_ID_CURR,'EXT_SOURCE_3']),ls='--' )







st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)







