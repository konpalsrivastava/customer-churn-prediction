import streamlit as st
import joblib
import pandas as pd
import numpy as np

#loading the trained model, the scaler, and the feature columns list
model = joblib.load('churn_model.pkl')
#model = joblib.load('C:/Users/KIIT0001/Desktop/customer_churn_project/churn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('feature_cols.pkl')

#defining the UI
st.title('Customer Churn Prediction App')
st.write('Enter customer information to get a churn prediction.')

#creating input fields for user data
gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
partner = st.selectbox('Partner', ['No', 'Yes'])
dependents = st.selectbox('Dependents', ['No', 'Yes'])


tenure = st.slider('Tenure (months)', 0, 72, 12)
monthly_charges = st.number_input('Monthly Charges', 0.0, 150.0, 50.0)
total_charges = st.number_input('Total Charges', 0.0, 10000.0, 500.0)


phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['No internet service', 'No', 'Yes'])
online_backup = st.selectbox('Online Backup', ['No internet service', 'No', 'Yes'])
device_protection = st.selectbox('Device Protection', ['No internet service', 'No', 'Yes'])
tech_support = st.selectbox('Tech Support', ['No internet service', 'No', 'Yes'])
streaming_tv = st.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'])
streaming_movies = st.selectbox('Streaming Movies', ['No internet service', 'No', 'Yes'])


contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

#creating a button for the prediction
if st.button('Predict Churn'):
    #creating a DataFrame from the user input
    user_data = {
        'gender': [gender], 'SeniorCitizen': [senior_citizen], 'Partner': [partner], 'Dependents': [dependents],
        'tenure': [tenure], 'PhoneService': [phone_service], 'MultipleLines': [multiple_lines],
        'InternetService': [internet_service], 'OnlineSecurity': [online_security], 'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection], 'TechSupport': [tech_support], 'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies], 'Contract': [contract], 'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method], 'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]
    }
    user_df = pd.DataFrame(user_data)

    #preprocessing the user data
    #1.handle TotalCharges for new customers
    if user_df['tenure'].iloc[0] == 0:
        user_df['TotalCharges'] = 0
    #2.replicating the one-hot encoding
    categorical_cols_app = user_df.select_dtypes(include='object').columns.tolist()
    user_df_encoded = pd.get_dummies(user_df, columns=categorical_cols_app)
    #3.reindexing the DataFrame to match the training data's columns
    user_df_final = user_df_encoded.reindex(columns=feature_cols, fill_value=0)
    #taking the user's preprocessed DataFrame and reordering its columns to perfectly match the order of the columns from our training data(feature_cols)
    #4.scaling the numerical data
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    user_df_final[numerical_cols] = scaler.transform(user_df_final[numerical_cols])

    #finally making the prediction
    prediction_probab = model.predict_proba(user_df_final)[:, 1]#will return the probability of a customer churning

   
    #and displaying the result
    st.subheader('Prediction Result:')
    st.write(f'The probability of this customer churning is: **{prediction_probab[0]:.2%}**')

    #setting a higher threshold value so now the model will only predict churn if it's at least 70% confident
    churn_probability = prediction_probab[0]
    if churn_probability >0.80:
      st.error('this customer is at HIGH RISK of churning.')
    elif churn_probability >0.50:
      st.warning('this customer is at MEDIUM RISK of churning')
    else:
      st.success('this customer is likely to be retained.')
