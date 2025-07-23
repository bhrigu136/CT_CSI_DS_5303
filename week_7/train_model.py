import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("train.csv")

# Select features and target
X = df[["OverallQual", "GrLivArea"]]
y = df["SalePrice"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print(" Model trained and saved.")
