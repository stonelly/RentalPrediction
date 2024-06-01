import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Load the trained model and bin edges
model = joblib.load('rentalPredictXgb.joblib')
bin_edges, mean_rent_by_location = joblib.load('bin_edges_and_mean_rent.pkl')

# Function to map location to bin
def map_location_to_bin(location, mean_rent_by_location, bin_edges):
    mean_rent = mean_rent_by_location.get(location)
    if mean_rent is not None:
        location_bin = pd.cut([mean_rent], bins=bin_edges, labels=range(1, 7))[0]
        return int(location_bin)
    else:
        st.error("Location not found in the database.")
        return None
    
# Convert 'Yes'/'No' to 1/0
def convert_yes_no(value):
    return 1 if value == 'Yes' else 0

# Convert furnished status to numerical value
def convert_furnished_status(value):
    if value == "Not Furnished":
        return 0
    elif value == "Partially Furnished":
        return 1
    elif value == "Fully Furnished":
        return 2
    
# Title and description
st.title('Monthly Rent Prediction in Klang Valley')
st.write('Enter the property details to predict the monthly rent.')

# Input fields
location = st.selectbox('Location', options=list(mean_rent_by_location.keys()))
property_type = st.selectbox('Property Type', ['Condominium', 'Apartment', 'Service Residence', 'Studio', 'Flat', 'Duplex', 'Others', 'Townhouse Condo'])
rooms = st.number_input('Number of Rooms', min_value=1, max_value=10, step=1)
size = st.number_input('Size (sqft)', min_value=300, max_value=3000, step=10)
furnished = st.selectbox('Furnished', ['Not Furnished', 'Partially Furnished', 'Fully Furnished'])
region = st.selectbox('Region', ['Kuala Lumpur', 'Selangor'])
gymnasium = st.selectbox('Gymnasium', ['Yes', 'No'])
air_cond = st.selectbox('Air-Conditioning', ['Yes', 'No'])
washing_machine = st.selectbox('Washing Machine', ['Yes', 'No'])
swimming_pool = st.selectbox('Swimming Pool', ['Yes', 'No'])

# Automatically determine the location bin
location_bin = map_location_to_bin(location, mean_rent_by_location, bin_edges)

if location_bin is not None:
    # Prediction
    if st.button('Predict'):
        
        # Convert 'Yes'/'No' inputs to 1/0
        gymnasium = convert_yes_no(gymnasium)
        air_cond = convert_yes_no(air_cond)
        washing_machine = convert_yes_no(washing_machine)
        swimming_pool = convert_yes_no(swimming_pool)
        furnished = convert_furnished_status(furnished)

        # Prepare the feature vector
        input_data = np.array([[location, property_type, rooms, size, furnished, region,
                                gymnasium, washing_machine, air_cond, swimming_pool, location_bin]])

        print("start predict....")
        print(input_data)

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # One-hot encode categorical variables
        input_df = pd.get_dummies(input_df, columns=['location', 'property_type', 'region'])
        
        print("start predict....") 
        # Make prediction
        prediction = model.predict(input_data)
        st.write(f'Predicted Monthly Rent: ${prediction[0]:.2f}')
