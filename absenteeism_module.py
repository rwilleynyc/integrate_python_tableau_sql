#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
 
# Create standard scaler for non-dummy variables

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # Initialize as Standard Scaler object
        self.scaler = StandardScaler(copy,with_mean,with_std)
        
        # Set columns to scale
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
        
    # Create fit method based on standard scaler
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    
    # Create transform method based on standard scaler
    def transform(self, X, y=None, copy=None):
        
        # Record initial order of columns
        init_col_order = X.columns
        
        # Scale all relevant features
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # Get unscaled data
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return dataframe with all features
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]



# Create class for predicting new data
class absenteeism_model():
       
        def __init__(self, model_file, scaler_file):
            # Load model and scaler
            with open('logreg_model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
         
        # Create function for preprocessing data
        def load_and_clean_data(self, data_file):
             
            # Import data
            df = pd.read_csv(data_file)
            
            # Store data as new variable for later use
            self.df_with_predictions = df.copy()
            
            # Replace spaces in column names with underscores
            df.columns = df.columns.str.replace(' ', '_')
            
            # Create absenteeism column with 'NaN' strings
            df['Absenteeism_Time_in_Hours'] = 'NaN'

            # Create reason groups
            reasons = {'Sickness': list(range(1, 15)), 
                       'Pregnancy': list(range(15, 18)), 
                       'Injury': list(range(18, 22)), 
                       'Doctors_Appt': list(range(22, 29))}

            # Get dummies for reasons
            for reason in reasons.keys():
                df[reason] = [1 if num in reasons[reason] else 0 for num in df['Reason_for_Absence']]

            # Drop Reason & ID columns
            df.drop(['Reason_for_Absence', 'ID'], axis=1, inplace=True)            
            
            # Convert date column to datetime format
            df.Date = pd.to_datetime(df.Date)

            # Get day name dummies
            df['Weekday'] = [date.weekday() for date in df.Date]
            df['Month'] = [month.month for month in df['Date']]

            df.drop('Date', axis=1, inplace=True)
 
            # 'Education' column to dummy indicating if hs completed
            df.Education = [1 if lvl > 1 else 0 for lvl in df.Education]
 
            # Fill N/A values with 0
            df = df.fillna(value=0)
 
            # Drop target variable
            df = df.drop(['Absenteeism_Time_in_Hours'],axis=1)
             
            # Save df to preprocessed variable
            self.preprocessed_data = df.copy()
             
            # Create scaled data variable for making predictions
            self.data = self.scaler.transform(df)


        # Predict probability of significant absences
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
         
        # Predict significant absences
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
         
        # Add probability and prediction columns to preprocessed data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data

