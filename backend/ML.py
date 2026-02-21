import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("medical_insurance.csv")

# Drop unnecessary column
df = df.drop(columns=['person_id'])

# Fix alcohol column
df['alcohol_freq'] = df['alcohol_freq'].fillna('Non-drinker')

# Select features
X = df[[
    "age",
    "sex",
    "bmi",
    "smoker",
    "income",
    "region",
    "claims_count",
    "annual_premium"
]]

y = df["annual_medical_cost"]

# Identify column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Pipelines
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

from sklearn.linear_model import Lasso
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso())
])

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# Save model
with open("insurance_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully as insurance_model.pkl")

pred=pipeline.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,pred))

