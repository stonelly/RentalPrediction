from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib

def preprocess_and_train_model(data):
    label_encoder = LabelEncoder()
    data['property_type'] = label_encoder.fit_transform(data['property_type'])
    data['region'] = label_encoder.fit_transform(data['region'])

    X = data.drop(['monthly_rent', 'location'], axis=1)
    y = data['monthly_rent']

    model = RandomForestRegressor()
    model.fit(X, y)

    return model

# Load data
data = pd.read_csv('cleaned_data.csv').copy()
model = preprocess_and_train_model(data)
# Save the trained model
joblib.dump(model, 'RF_model.pkl')