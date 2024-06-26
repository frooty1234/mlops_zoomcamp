#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('python -V')


# In[2]:


#get_ipython().system('pip freeze | grep scikit-learn')


# In[8]:


import os
import uuid
import pickle
import sys

import pandas as pd


from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


# In[9]:


#year = 2023
#month = 3
#taxi_type = 'yellow'

#input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
#output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
#output_file = f'{year:04d}-{month:02d}.parquet'

#model_path = 'model.bin'
#RUN_ID = os.getenv('RUN_ID', 'e1efc53e9bd149078b0c12aeaa6365df')


# In[10]:


def read_dataframe(filename: str, year, month):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    #df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    #categorical = ['PU_DO']
    #numerical = ['trip_distance']
    #dicts = df[categorical + numerical].to_dict(orient='records')
    dicts = df[categorical].to_dict(orient='records')
    return dicts


# In[15]:


def load_model(model_path):
    #logged_model = RUN_ID
    #model = mlflow.pyfunc.load_model(logged_model)
    with open(model_path, 'rb') as f_in:
        dv,model = pickle.load(f_in)
    return dv, model


def apply_model(input_file, model_path, output_file, year, month):
    df = read_dataframe(input_file, year, month)
    dicts = prepare_dictionaries(df)

    
    dv, model = load_model(model_path)
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    #y_pred = model.predict(dicts)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    #df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    #df_result['PULocationID'] = df['PULocationID']
    #df_result['DOLocationID'] = df['DOLocationID']
    #df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    #df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    #df_result['model_version'] = run_id
    
    df_result.to_parquet(output_file , engine='pyarrow', compression=None, index=False)
    print(f"Predicted mean duration {y_pred.mean()}")


# In[16]:


#apply_model(input_file=input_file, model_path=model_path, output_file=output_file)


# In[17]:


#get_ipython().system('ls')


# In[18]:


#get_ipython().system('ls -hl 2023-03.parquet')


# In[ ]:


def run():
    #year = 2021
    #month = 3
    if len(sys.argv) < 2:
        print("[!] Please specify year and month")
        print("SYNTAX: python score.py [YEAR] [MONTH]")
        sys.exit(1)

    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3

    #RUN_ID = '54c0663c677e417f8900b51ed7985878'
    #run_id = sys.argv[3] 

    #input_file = '../../data/green_tripdata_2021-01.parquet'
    taxi_type = 'yellow'

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    output_file = f'{year:04d}-{month:02d}.parquet'

    model_path = 'model.bin'
    #input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_{year:04}-{month:02d}.parquet'
    #output_file = f'../../output/green_tripdata_{year:04}-{month:02d}.parquet'


    apply_model(input_file=input_file, model_path =model_path, output_file=output_file, year = year, month = month)

if __name__ == '__main__':
    run()

