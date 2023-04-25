import streamlit as st
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#loading dataset
df=pd.read_csv(r'C:\Users\prati\OneDrive\Desktop\mlproject\src\google colab\data\data_mobile_price_range.csv')

#Feature manipulation
df['px_height'] = df['px_height'].replace(0,df['px_height'].mean())

df['sc_w'] = df['sc_w'].replace(0,df['sc_w'].mean())

#Feature Engineering
df['screen_area'] = df['sc_h'] * df['sc_w']
df['px_area'] = df['px_height'] * df['px_width']

#Feature selection
X = df[['screen_area','n_cores','int_memory','mobile_wt','px_area','battery_power','ram']]
y = df['price_range']

#Preprocessing using powertransformer
pt=preprocessing.PowerTransformer(copy=False)
X[['screen_area','px_area']]=pt.fit_transform(X[['screen_area','px_area']])


#Scaling data using standard scaler
scaler = StandardScaler()

#Fit on data
X_scaled=scaler.fit_transform(X)
X_scaled=pd.DataFrame(data=X_scaled,columns=X.columns,index=X.index)



st.title(':blue[Mobile Price Range Predictor]')
st.subheader(':green[Enter accurate information to predict price range]' )

#Attributes regarding system
st.subheader(':red[Enter sytem information]') 
col1, col2, col3, col4= st.columns(4)



with col1:
  n_cores= st.slider(':green[No. of cores]',1,8)
  st.write('No. of cores:', n_cores)

with col2:
    ram= st.slider(':green[Select Ram in MB]',256, 4096)
    st.write('Ram in MB:', ram)

with col3:
  int_mem= st.slider(':green[Internal Memory]',2,64)
  st.write('Internal Memory:', int_mem)    

with col4:
  mob_weight= st.slider(':green[Mobile Weight]',80,200)
  st.write('[mobile weight:', mob_weight)






    

st.subheader(':red[Enter Screen attributes]') 


#Attributes regarding screen
col1, col2, col3, col4= st.columns(4)  


with col1:
  sc_height= st.slider(':green[Select height in cm]',12, 19)
  st.write('Screen Height:',sc_height)
   

with col2:
  sc_wid= st.slider(':green[Select Width in cm]',5, 18)
  st.write('Screen Width:',sc_wid)

with col3:
  px_height= st.slider(':green[Pixel resolution height]',100, 1960)
  st.write('Pixel Heigth is:',px_height)
   

with col4:
  px_width= st.slider(':green[Pixel resolution width]',500, 2000)
  st.write('Pixel width is:',px_width)
  

st.subheader(':red[Select Battery power in mAh]')
battery_pow= st.slider(':green[Battery Power]',501, 2000)
st.write('Selected battery power is:',battery_pow)


 

#Taking input and predicting using saved model 
if st.button('Predict'):
    dict={'screen_area':sc_height*sc_wid, 'n_cores':n_cores,'int_memory':int_mem,'mobile_wt':mob_weight,
      'px_area':px_height*px_width,'battery_power':battery_pow,'ram':ram,}
      
    #converting input to dataframe
    df=pd.DataFrame([dict])
    
    
    #Preproceesing data
    df[['screen_area','px_area']]=pt.transform(df[['screen_area','px_area']])
    #transformation on data
    df=scaler.transform(df)

    #Loading saved model
    loaded_model = pickle.load(open(r'C:\Users\prati\OneDrive\Desktop\mlproject\mobile_price_predict_new.sav', 'rb'))
    y_preds=loaded_model.predict(df)
    if (y_preds==0):
      st.subheader('Selected phone is low cost phone')
    elif(y_preds==1):
      st.subheader('Selected phone is medium cost phone')
    elif(y_preds==2):
      st.subheader('Selected phone is high price phone')
    else:
      st.subheader('Selected phone is very high price phone')    
        



























