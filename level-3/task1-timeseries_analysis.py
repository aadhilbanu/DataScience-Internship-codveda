import pandas as pd
import matplotlib.pyplot as plt

# Create sample sales data
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Sales": [100, 120, 130, 125, 150, 170]
}

df = pd.DataFrame(data)

print(df)

# Plot sales trend
plt.plot(df["Month"], df["Sales"], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# Moving Average
df["Moving_Avg"] = df["Sales"].rolling(window=2).mean()

print("\nWith Moving Average:")
print(df)

# Plot moving average
plt.plot(df["Month"], df["Sales"], label="Sales", marker='o')
plt.plot(df["Month"], df["Moving_Avg"], label="Moving Average", marker='o')

plt.title("Sales Forecasting")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
