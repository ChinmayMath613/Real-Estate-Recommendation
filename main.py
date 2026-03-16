import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

cleaned_df = pd.read_csv('pune_cleaned_data.csv')
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
    
    return cleaned_df.iloc[indices[0]]

def search_properties(min_price=None, max_price=None, location=None, 
                      developer=None, sort_by='price', ascending=True):
    results = cleaned_df.copy()

    if min_price: results = results[results['price'] >= min_price]
    if max_price: results = results[results['price'] <= max_price]
    if location: results = results[results['locality'].str.contains(location, case=False, na=False)]
    if developer: results = results[results['projectname'].str.contains(developer, case=False, na=False)]

    sort_map = {'price': 'price', 'rate_sqft': 'pricepersquare', 'possession': 'possesiondate'}
    column = sort_map.get(sort_by, 'price')
    
    return results.sort_values(by=column, ascending=ascending)

def compare_listings(indices):
    """Compares up to 3 listings side-by-side including unique amenities."""
    selected = cleaned_df.iloc[indices].copy()
    
    specs = ['projectname', 'locality', 'price', 'pricepersquare', 'area', 'bhk', 'possesiondate', 'status']
    base_comparison = selected[specs].set_index('projectname').T

    amenity_sets = []
    for amenities in selected['amenitiesavailable']:
        s = set(item.strip() for item in str(amenities).split(',') if item.strip())
        amenity_sets.append(s)

    if amenity_sets:
        common = set.intersection(*amenity_sets)
    else:
        common = set()

    unique_extras = {}
    for i, project_name in enumerate(selected['projectname']):
        extras = amenity_sets[i] - common
        unique_extras[project_name] = ", ".join(list(extras)[:5]) + "..." 

    extras_row = pd.Series(unique_extras, name='Extra Amenities')
    final_comparison = pd.concat([base_comparison, extras_row.to_frame().T])

    return final_comparison, list(common)[:10] 

comparison_table, shared_perks = compare_listings([0, 1, 2])

print(comparison_table)
print(f"\nCommon Features across all: {', '.join(shared_perks)}")