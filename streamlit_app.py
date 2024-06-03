import joblib
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Page title
st.set_page_config(page_title='Rental Price Prediction')
st.title('Rental Price Prediction')
st.info('This is a rental price prediction mainly based in KL & Selangor area. Fill in the information and check the predicted price below.')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv').copy()  # Make a copy of the DataFrame to avoid mutation

# Function to map 'Yes' or 'No' to 1 or 0
def map_yes_no_to_binary(value):
    return 1 if value == 'Yes' else 0

# Define region bins
region_bins = {
    'Kuala Lumpur': 0,
    'Selangor': 1
}

# Define furnished bins
furnished_bins = {
    'Not Furnished': 0,
    'Partially Furnished': 1,
    'Fully Furnished': 2
}

# Define a dictionary that maps each location to its corresponding region
location_to_region = {
    'Cheras': 'Selangor',
    'Taman Desa': 'Kuala Lumpur',
    'Sentul': 'Kuala Lumpur',
    'Mont Kiara': 'Kuala Lumpur',
    'Setapak': 'Kuala Lumpur',
    'Ampang': 'Selangor',
    'Segambut': 'Kuala Lumpur',
    'Desa ParkCity': 'Kuala Lumpur',
    'Bukit Jalil': 'Kuala Lumpur',
    'Kepong': 'Kuala Lumpur',
    'Wangsa Maju': 'Kuala Lumpur',
    'Jalan Kuching': 'Kuala Lumpur',
    'Bandar Menjalara': 'Kuala Lumpur',
    'Old Klang Road': 'Kuala Lumpur',
    'Desa Pandan': 'Kuala Lumpur',
    'KLCC': 'Kuala Lumpur',
    'Ampang Hilir': 'Kuala Lumpur',
    'Bukit Bintang': 'Kuala Lumpur',
    'KL City': 'Kuala Lumpur',
    'Jalan Ipoh': 'Kuala Lumpur',
    'Setiawangsa': 'Kuala Lumpur',
    'Gombak': 'Selangor',
    'Sungai Besi': 'Kuala Lumpur',
    'Jinjang': 'Kuala Lumpur',
    'Sri Petaling': 'Kuala Lumpur',
    'Bangsar South': 'Kuala Lumpur',
    'Pantai': 'Kuala Lumpur',
    'Brickfields': 'Kuala Lumpur',
    'Kuchai Lama': 'Kuala Lumpur',
    'Jalan Sultan Ismail': 'Kuala Lumpur',
    'Bangsar': 'Kuala Lumpur',
    'Pandan Indah': 'Kuala Lumpur',
    'Pandan Jaya': 'Kuala Lumpur',
    'Damansara Heights': 'Kuala Lumpur',
    'Bandar Damai Perdana': 'Kuala Lumpur',
    'Titiwangsa': 'Kuala Lumpur',
    'Bandar Tasik Selatan': 'Kuala Lumpur',
    'Pandan Perdana': 'Kuala Lumpur',
    'Keramat': 'Kuala Lumpur',
    'Pudu': 'Kuala Lumpur',
    'OUG': 'Kuala Lumpur',
    'Taman Tun Dr Ismail': 'Kuala Lumpur',
    'Sri Hartamas': 'Kuala Lumpur',
    'Solaris Dutamas': 'Kuala Lumpur',
    'Puchong': 'Selangor',
    'Seputeh': 'Kuala Lumpur',
    'Sri Damansara': 'Kuala Lumpur',
    'Taman Melawati': 'Kuala Lumpur',
    'Desa Petaling': 'Kuala Lumpur',
    'Others': 'Kuala Lumpur',
    'Serdang': 'Selangor',
    'City Centre': 'Kuala Lumpur',
    'Salak Selatan': 'Kuala Lumpur',
    'Sungai Penchala': 'Kuala Lumpur',
    'Mid Valley City': 'Kuala Lumpur',
    'Damansara': 'Kuala Lumpur',
    'Cyberjaya': 'Selangor',
    'Shah Alam': 'Selangor',
    'Klang': 'Selangor',
    'Petaling Jaya': 'Selangor',
    'Subang Jaya': 'Selangor',
    'Bandar Sunway': 'Selangor',
    'Seri Kembangan': 'Selangor',
    'Kajang': 'Selangor',
    'Rawang': 'Selangor',
    'Kota Damansara': 'Selangor',
    'Batu Caves': 'Selangor',
    'Semenyih': 'Selangor',
    'Bukit Jelutong': 'Selangor',
    'USJ': 'Selangor',
    'Damansara Damai': 'Selangor',
    'Bandar Mahkota Cheras': 'Selangor',
    'Puncak Alam': 'Selangor',
    'Sepang': 'Selangor',
    'Kuala Langat': 'Selangor',
    'Setia Alam': 'Selangor',
    'Selayang': 'Selangor',
    'Sungai Buloh': 'Selangor',
    'Bangi': 'Selangor',
    'Dengkil': 'Selangor',
    'Ara Damansara': 'Selangor',
    'I-City': 'Selangor',
    'Bandar Sri Damansara': 'Selangor',
    'Damansara Perdana': 'Selangor',
    'Bandar Saujana Putra': 'Selangor',
    'Kota Kemuning': 'Selangor',
    'Ulu Klang': 'Selangor',
    'Kapar': 'Selangor',
    'Balakong': 'Selangor',
    'Bandar Sungai Long': 'Selangor',
    'Port Klang': 'Selangor',
    'Hulu Langat': 'Selangor',
    'Bandar Kinrara': 'Selangor',
    'Jenjarom': 'Selangor',
    'Glenmarie': 'Selangor',
    'Kelana Jaya': 'Selangor',
    'Puchong South': 'Selangor',
    'Alam Impian': 'Selangor',
    'Pulau Indah (Pulau Lumut)': 'Selangor',
    'Bandar Bukit Tinggi': 'Selangor',
    'Putra Heights': 'Selangor',
    'Saujana Utama': 'Selangor',
    'Bandar Bukit Raja': 'Selangor',
    'Bandar Utama': 'Selangor',
    'Subang Bestari': 'Selangor',
    'Bandar Botanic': 'Selangor',
    'Banting': 'Selangor',
    'Kuala Selangor': 'Selangor',
    'Salak Tinggi': 'Selangor',
    'Serendah': 'Selangor',
    'Bukit Beruntung': 'Selangor',
    'Mutiara Damansara': 'Selangor',
    'Telok Panglima Garang': 'Selangor',
    'Bukit Subang': 'Selangor',
    'Puncak Jalil': 'Selangor'
}

# Define location bins
location_bins = {
    'Alam Impian': 5,
    'Ampang': 3,
    'Ampang Hilir': 6,
    'Ara Damansara': 6,
    'Balakong': 2,
    'Bandar Botanic': 2,
    'Bandar Bukit Raja': 3,
    'Bandar Bukit Tinggi': 6,
    'Bandar Damai Perdana': 4,
    'Bandar Kinrara': 3,
    'Bandar Mahkota Cheras': 4,
    'Bandar Menjalara': 4,
    'Bandar Saujana Putra': 3,
    'Bandar Sri Damansara': 1,
    'Bandar Sungai Long': 1,
    'Bandar Sunway': 4,
    'Bandar Tasik Selatan': 1,
    'Bandar Utama': 4,
    'Bangi': 2,
    'Bangsar': 6,
    'Bangsar South': 5,
    'Banting': 1,
    'Batu Caves': 2,
    'Brickfields': 6,
    'Bukit Beruntung': 1,
    'Bukit Bintang': 6,
    'Bukit Jalil': 3,
    'Bukit Jelutong': 5,
    'Bukit Subang': 1,
    'Cheras': 3,
    'City Centre': 6,
    'Cyberjaya': 3,
    'Damansara': 6,
    'Damansara Damai': 2,
    'Damansara Heights': 4,
    'Damansara Perdana': 3,
    'Dengkil': 2,
    'Desa Pandan': 5,
    'Desa ParkCity': 6,
    'Desa Petaling': 2,
    'Glenmarie': 5,
    'Gombak': 4,
    'Hulu Langat': 1,
    'I-City': 4,
    'Jalan Ipoh': 5,
    'Jalan Kuching': 5,
    'Jalan Sultan Ismail': 6,
    'Jenjarom': 2,
    'Jinjang': 1,
    'KL City': 6,
    'KLCC': 6,
    'Kajang': 2,
    'Kapar': 1,
    'Kelana Jaya': 5,
    'Kepong': 3,
    'Keramat': 5,
    'Klang': 3,
    'Kota Damansara': 5,
    'Kota Kemuning': 6,
    'Kuala Langat': 3,
    'Kuala Selangor': 1,
    'Kuchai Lama': 4,
    'Mid Valley City': 6,
    'Mont Kiara': 6,
    'Mutiara Damansara': 2,
    'OUG': 3,
    'Old Klang Road': 5,
    'Others': 3,
    'Pandan Indah': 2,
    'Pandan Jaya': 2,
    'Pandan Perdana': 4,
    'Pantai': 4,
    'Petaling Jaya': 3,
    'Port Klang': 4,
    'Puchong': 2,
    'Puchong South': 2,
    'Pudu': 5,
    'Pulau Indah (Pulau Lumut)': 2,
    'Puncak Alam': 1,
    'Puncak Jalil': 1,
    'Putra Heights': 2,
    'Rawang': 1,
    'Salak Selatan': 1,
    'Salak Tinggi': 1,
    'Saujana Utama': 2,
    'Segambut': 5,
    'Selayang': 3,
    'Semenyih': 1,
    'Sentul': 4,
    'Sepang': 2,
    'Seputeh': 5,
    'Serdang': 1,
    'Serendah': 1,
    'Seri Kembangan': 2,
    'Setapak': 4,
    'Setia Alam': 4,
    'Setiawangsa': 5,
    'Shah Alam': 3,
    'Solaris Dutamas': 5,
    'Sri Damansara': 6,
    'Sri Hartamas': 5,
    'Sri Petaling': 4,
    'Subang Bestari': 2,
    'Subang Jaya': 3,
    'Sungai Besi': 4,
    'Sungai Buloh': 3,
    'Sungai Penchala': 1,
    'Taman Desa': 4,
    'Taman Melawati': 6,
    'Taman Tun Dr Ismail': 6,
    'Telok Panglima Garang': 5,
    'Titiwangsa': 6,
    'USJ': 3,
    'Ulu Klang': 6,
    'Wangsa Maju': 4
}

data = load_data()
label_encoder = LabelEncoder()
label_encoder.fit(data['property_type'])

# Train the model
model = joblib.load('RF_model.pkl')

# Rental Price prediction form
with st.form('predict'):
    location = st.selectbox('Location', list(location_bins.keys()))
    region = st.selectbox('Region', list(region_bins.keys()))
    property_type = st.selectbox('Property Type',['Apartment', 'Condominium', 'Duplex' ,'Flat', 'Service Residence', 'Studio', 'Townhouse Condo', 'Others'])
    rooms = st.selectbox('Rooms Number',['1', '2','3','4','5','6','7','8','9','10'])
    size = st.number_input('Size (sqft)')
    furnished = st.selectbox('Furnished',list(furnished_bins.keys()))
    gymnasium = st.radio("Gymnasium", ("Yes", "No"))
    air_cond = st.radio("Air-cond", ("Yes", "No"))
    washing_machine = st.radio("Washing Machine", ("Yes", "No"))
    swimming_pool = st.radio("Swimming Pool", ("Yes", "No"))
    submit = st.form_submit_button('Predict')

if submit:
    property_type = int(label_encoder.transform([property_type])[0])
    rooms = int(rooms)
    gymnasium = map_yes_no_to_binary(gymnasium)
    air_cond = map_yes_no_to_binary(air_cond)
    washing_machine = map_yes_no_to_binary(washing_machine)
    swimming_pool = map_yes_no_to_binary(swimming_pool)
    location_bin = location_bins[location]
    furnished = furnished_bins[furnished]
    region = region_bins[region]
    input_data = [[property_type, rooms, size, furnished, region, gymnasium, air_cond, washing_machine, swimming_pool, location_bin]]
    prediction = model.predict(input_data)
    st.write("Predicted Rental Price:", round(prediction[0], 2))
