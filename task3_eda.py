import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_iris.csv")

# Summary statistics
print("Summary Statistics:")
print(df.describe())

# Count plot
sns.countplot(x="species", data=df)
plt.title("Species Count")
plt.show()

# Pairplot
sns.pairplot(df, hue="species")
plt.show()

# Histogram
df.hist(figsize=(10,8))
plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()
