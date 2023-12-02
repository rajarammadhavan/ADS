import streamlit as st
import datetime
import pandas as pd 
import numpy as np 

#library imports
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score, precision_score, roc_auc_score, recall_score,roc_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from time import time




#Multiple page configurations
st.set_page_config(
    page_title="Credit Card Eligibility", #Main.py file as main page
    page_icon="💳",
)

#Background image (WIP)
# st.markdown("""
#     <style>
#         .stApp {
#         background: url("");
#         background-size: cover;
#         }
#     </style>""", unsafe_allow_html=True)

st.title("Prediction of Credit Card approval")

st.sidebar.info("You are now on the eligibility page ✅")

st.write("""
        ### Check if you are eligible in seconds!✅💳
""")

#Get input from user
def input_features():

        #gender
        GENDER = st.selectbox("Select your Gender",("M","F"),index=None,placeholder="Select your option")

        #Birthday_count
        b_day = st.date_input("Your birthday date", min_value = datetime.date(1950,1,1))
        Birthday_count = np.abs((b_day - datetime.date.today()))
        Birthday_count = Birthday_count.days

        #Marital_status
        Marital_status = st.selectbox("Select your Marital status",("Single / not married","Married","Civil marriage","Separated","Widow"),index=None,placeholder="Select your option")

        #Children
        CHILDREN = st.slider("How many dependent children are currently under your care or support?",0,14)

        #Family_Members
        Family_Members = st.slider("How many number of family members?",0,15)


        #EDUCATION
        EDUCATION = st.selectbox("Select your Education level",("Lower secondary","Secondary / secondary special","Higher education","Academic degree","Incomplete higher"),index=None,placeholder="Select your option")

        #Employed_days
        employment_choice = st.selectbox('Choose your current employment status',('Employed','Unemployed'))
        min_date = datetime.date(1950,1,1)
        max_date = datetime.date.today()
        if employment_choice == 'Employed':
                #Employed_days
                em = st.date_input("Select your most recent employment date", min_value = min_date,max_value = max_date)
                Employed_days = (em - datetime.date.today())
                Employed_days = Employed_days.days
        if employment_choice == 'Unemployed':
                em = st.date_input("Select your the daterange of your unemployment", min_value = min_date,max_value=max_date)
                Employed_days = (datetime.date.today()- em)
                Employed_days = Employed_days.days

        #Type_Occupation
        Type_Occupation = st.selectbox("Select your Occupation type",("Laborers","Core staff","Managers","Sales staff","Drivers","High skill tech staff","Medicine staff","Accountants","Security staff","Cleaning staff","Cooking staff","Private service staff","Secretaries","Low-skill Laborers","Waiters/barmen staff","HR staff","IT staff","Realty agents"),index=None,placeholder="Select your option")

        #Annual_income
        Annual_income = st.slider("What is your total yearly earnings?",30000,1500000,step=5000)

        #Type_Income
        Type_Income = st.selectbox("Select your type of Income",("State servant","Pensioner","Commercial associate","Working"),index=None,placeholder="Select your option")

        #Car_owner
        Car_Owner = st.selectbox("Do you own ateast one car?",("Y","N"),index=None,placeholder="Select your option")

        #Propert_Owner
        Propert_Owner = st.selectbox("Do you own ateast one property?",("Y","N"),index=None,placeholder="Select your option")

        #Housing_type
        Housing_type = st.selectbox("Select your Housing type",("House / apartment","With parents","Rented apartment","Municipal apartment","Co-op apartment","Office apartment"),index=None,placeholder="Select your option")

        #Mobile_phone
        Mobile_phone = st.selectbox("Do you own a Mobile phone?",("Y","N"),index=None,placeholder="Select your option")


        #Work_Phone
        Work_Phone = st.selectbox("Do you own a Work phone?",("Y","N"),index=None,placeholder="Select your option")


        #Phone
        Phone = st.selectbox("Do you own atleast one phone number?",("Y","N"),index=None,placeholder="Select your option")


        #EMAIL_ID
        EMAIL_ID = st.selectbox("Do you have ateast one email ID created?",("Y","N"),index=None,placeholder="Select your option")


        # days to year conversion
        d = Birthday_count/365
        em_conv = Employed_days/365

        data = { 
                'GENDER':GENDER,
                'Car_Owner':Car_Owner,
                'Propert_Owner':Propert_Owner,
                'CHILDREN':CHILDREN,
                'Annual_income':Annual_income,
                'Type_Income':Type_Income,
                'EDUCATION':EDUCATION,
                'Marital_status': Marital_status,    
                'Housing_type'  : Housing_type,
                'Birthday_count'   :Birthday_count,
                'Employed_days'      :Employed_days,
                'Mobile_phone'       : Mobile_phone,
                'Work_Phone'         :Work_Phone,
                'Phone'              :Phone,
                'EMAIL_ID'           :EMAIL_ID,
                'Type_Occupation'    :Type_Occupation,
                'Family_Members' :Family_Members,
                'Age_conv': d,
                'Employed years': em_conv
                }
                                        
        user_input = pd.DataFrame(data,index=[0])
        return user_input

#Create a new dataframe for user input features
df = input_features()

zip_file = "Credit_card.zip"

# opening the zip file in READ mode
with ZipFile(zip_file, 'r') as zip:
    zip.printdir()
    print('Extracting...')
    zip.extractall()
    print('Extracted successfully!')


#load data
cc_df    = pd.read_csv("Credit_card.csv")
cc_label = pd.read_csv("Credit_card_label.csv")

cc_final = pd.merge(cc_df, cc_label, on='Ind_ID')

pre_cc = cc_final

pre_cc['Annual_income'].fillna(pre_cc['Annual_income'].median(),inplace =True)
pre_cc['Birthday_count'].fillna(pre_cc['Birthday_count'].mean(),inplace =True)

pre_cc['GENDER'].fillna(pre_cc['GENDER'].mode()[0],inplace =True)
pre_cc.dropna(subset=['Type_Occupation'], inplace=True)

pre_cc['Age_conv'] = np.abs(pre_cc['Birthday_count'])/365

#converting employed days into years
pre_cc['Employed_years'] = (pre_cc['Employed_days']/365)

pre_cc = pre_cc.reset_index(drop=True)

ohe = OneHotEncoder(sparse=False,handle_unknown='ignore')
cat_attribs = ['GENDER','Car_Owner','Propert_Owner','Type_Income','EDUCATION','Marital_status','Housing_type','Type_Occupation']
enc_df = pd.DataFrame(ohe.fit_transform(pre_cc[['GENDER','Car_Owner','Propert_Owner','Type_Income','EDUCATION','Marital_status','Housing_type','Type_Occupation']]), columns = ohe.get_feature_names_out())

cc_df = pre_cc.join(enc_df)

cc_df.drop(cat_attribs,axis=1,inplace=True)

X = cc_df.drop(['Ind_ID','label'],axis = 'columns')
y = cc_df['label']

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,stratify = y,random_state = 42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.random.seed(42)
import json

rfc = RandomForestClassifier(class_weight='balanced', random_state=42)

params_grid = {
            'max_depth': [5,10,15],
            'max_features': [10,15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1,3],
            'bootstrap': [True],
            'n_estimators':[25,50,100]
          }

rfc_grid = GridSearchCV(rfc, params_grid,scoring='f1',cv =5, n_jobs = -1)
rfc_grid.fit(X_train,y_train)

best_params = rfc_grid.best_estimator_.get_params()

param_dump = []
for i in sorted(params_grid):
  param_dump.append((i, best_params[i]))


# start = time()
rfc_model = rfc_grid.best_estimator_.fit(X_train,y_train)

cat_attribs1 = ['GENDER','Car_Owner','Propert_Owner','Type_Income','EDUCATION','Marital_status','Housing_type','Type_Occupation']
enc_df1 = pd.DataFrame(ohe.fit_transform(df[['GENDER','Car_Owner','Propert_Owner','Type_Income','EDUCATION','Marital_status','Housing_type','Type_Occupation']]), columns = ohe.get_feature_names_out())

cc_df1 = pre_cc.join(enc_df1)

cc_df1.drop(cat_attribs1,axis=1,inplace=True)

prediction = rfc_model.predict(cc_df1)
st.write(prediction)

#Prediction button
if st.button('Check my approval'):
        #will improve this section once the model part is ready,
        # until then this is just a example message when you click the button for prediction
     if prediction == 0:
        st.success('You are not eligible')
     if prediction == 1:
        st.success('You are  eligible')
     else:
        st.success('You are dummy')