from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import joblib

# Define the hyperparameters grid
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

def preprocess_and_train_model(data):
    label_encoder = LabelEncoder()
    data['property_type'] = label_encoder.fit_transform(data['property_type'])
    data['region'] = label_encoder.fit_transform(data['region'])

    X = data.drop(['monthly_rent', 'location'], axis=1)
    y = data['monthly_rent']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Define the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    model = XGBRegressor()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply the preprocessor and fit on training data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Perform Grid Search with 5-fold cross-validation for MSE
    grid_search = GridSearchCV(estimator=model, param_grid=xgb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Train the model with the best hyperparameters
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    return model

# Load data
data = pd.read_csv('cleaned_data.csv').copy()
model = preprocess_and_train_model(data)
# Save the trained model
joblib.dump(model, 'v2_rental_prediction_model.pkl')