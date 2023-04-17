import pickle
import pandas as pd
import streamlit as st
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from lightgbm import LGBMRegressor


#Loading dataset
df=pd.read_csv(r"C:\Users\prati\OneDrive\Desktop\mlproject\src\google colab\default of credit card clients.csv",skiprows = 1)

st.title('Credit Card Default Prediction')


#Taking stores number as input
cust_id = st.number_input('Enter customer ID',min_value=1, max_value=30000)

st.write('Customer ID selected is ',cust_id)



#Function to process input dataframe
def data_predict(df,cust_id):

        #converting all others values on education to 4
        df['EDUCATION']=df['EDUCATION'].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
        #Converting others in marriage to 3
        df['MARRIAGE']=df['MARRIAGE'].map({0:3,1:1,2:2,3:3})

        #Selected features
        numeric_col = ['LIMIT_BAL', 'AGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        
       
        # Transform Your data
        pt=preprocessing.PowerTransformer(copy=False)
        df[numeric_col]=pt.fit_transform(df[numeric_col])

        # Scaling your data
        X=df.drop(['default payment next month','ID'],axis=1)
        #Creating object 
        X_scaler = StandardScaler()
        #Fit on data
        X_scaled=X_scaler.fit_transform(X)

        #Converting to dataframe
        X_scaled_df=pd.DataFrame(data=X_scaled,columns=X.columns,index=X.index)
        
        #Adding customer ID on dataset
        X_scaled_df['ID']=df['ID']

        #Taking selected input from dataset
        input_df=X_scaled_df[(X_scaled_df["ID"]==cust_id)]
        input_df=input_df.drop(['ID'],axis=1)



        # Load the model File and predicting on unseen data.
        loaded_model = pickle.load(open(r'C:\Users\prati\OneDrive\Desktop\mlproject\credit_card_default.sav', 'rb'))
        default_prediction= loaded_model.predict(input_df)

        if (default_prediction==1):
            return 'There is high probablity that selected customer might default'
        else:
            return 'Selected customer is higly unlikely to default'   

        
#Caling function using predict button
if st.button('Default Prediction'):
    default_predict=data_predict(df,cust_id)
    st.write(default_predict)     


