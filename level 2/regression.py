import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("iris.csv")

# Convert species to numeric
df['species'] = df['species'].astype('category').cat.codes

# Features & target
X = df.drop("species", axis=1)
y = df["species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Score
score = model.score(X_test, y_test)

print("Regression Model Score:", score)
