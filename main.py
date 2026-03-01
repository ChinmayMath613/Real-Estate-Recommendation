import pandas as pd    
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

pd.set_option('display.max_columns', None)

df = pd.read_csv('Pune_property_data.csv', index_col=0)

def get_number(value):
    if pd.isna(value) or value == "":
        return None
    nums = re.findall(r"(\d+\.?\d*)", str(value).replace(',', ''))
    return float(nums[0]) if nums else None

df['area'] = df['area'].apply(get_number)

df['bhk'] = df['bhk'].apply(get_number)

df['bathroom'] = pd.to_numeric(df['bathroom'], errors='coerce')
df['balconies'] = pd.to_numeric(df['balconies'], errors='coerce')

df['price'] = pd.to_numeric(df['price'], errors='coerce')

features = ['price', 'area', 'bathroom', 'bhk', 'balconies']
df[features] = df[features].fillna(0)

numerical_data = df[features]

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numerical_data)

normalized_df = pd.DataFrame(normalized_data, columns=features)

def get_recommendations(user_prefs_raw, user_weights):
    weights_arr = np.array(user_weights)
    
    current_weighted_data = normalized_df * weights_arr 
    
    model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    model.fit(current_weighted_data)
    
    user_input_df = pd.DataFrame([user_prefs_raw], columns=features)
    user_normalized = scaler.transform(user_input_df)

    weighted_user_vals = user_normalized * weights_arr
    
    weighted_user_df = pd.DataFrame(weighted_user_vals, columns=features)
    
    distances, indices = model.kneighbors(weighted_user_df)
    return indices[0]

recommendation_indices = get_recommendations([10000000, 1200, 2, 2, 1], [5, 3, 3, 3, 1])

recommendations = df.iloc[recommendation_indices]

print(f"Top 5 Recommendation: \n{recommendations}")