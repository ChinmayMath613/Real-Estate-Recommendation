import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('Pune_property_data.csv') 
normalized_df = pd.read_csv('pune_normalized_data.csv') 
scaler = joblib.load('property_scaler.pkl') 

features = ['price', 'area', 'bathroom', 'bhk', 'balconies']

def get_recommendations(user_prefs_raw, user_weights):
    weights_arr = np.array(user_weights)
    
    current_weighted_data = normalized_df * weights_arr 
    
    model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    model.fit(current_weighted_data)
    
    user_input_df = pd.DataFrame([user_prefs_raw], columns=features)
    user_normalized = scaler.transform(user_input_df)
    
    weighted_user_df = pd.DataFrame(user_normalized * weights_arr, columns=features)
    distances, indices = model.kneighbors(weighted_user_df)
    
    return indices[0]

idx = get_recommendations([10000000, 1200, 2, 2, 1], [5, 3, 3, 3, 1])
print("Recommended Properties:")
print(df.iloc[idx][['price', 'area', 'bathroom', 'balconies']])