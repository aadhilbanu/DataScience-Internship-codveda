import pandas as pd

# Load dataset
df = pd.read_csv("iris.csv")

print("Original Data:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_iris.csv", index=False)

print("\nData Cleaning Completed!")
print(df.head())
