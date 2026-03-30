# ================================
# Sales Data Cleaning & Analysis
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("data/sales_data.csv")

print("Initial Data Shape:", df.shape)
print(df.head())

# -------------------------------z``
# 2. Data Understanding
# -------------------------------
print("\nData Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# 3. Data Cleaning
# -------------------------------

# Fill missing customer names
df["Customer_Name"].fillna("Unknown Customer", inplace=True)

# Fill missing Quantity with median
df["Quantity"].fillna(df["Quantity"].median(), inplace=True)

# Fill missing Price with median
df["Price"].fillna(df["Price"].median(), inplace=True)

# Convert Order_Date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Remove duplicate records
df.drop_duplicates(inplace=True)

print("\nAfter Cleaning:")
print(df.isnull().sum())
print("Cleaned Data Shape:", df.shape)

# -------------------------------
# 4. Feature Engineering
# -------------------------------
df["Total_Sales"] = df["Quantity"] * df["Price"]

# Extract month
df["Month"] = df["Order_Date"].dt.month

print(df.head())

# -------------------------------
# 5. Exploratory Data Analysis
# -------------------------------

# Total revenue
total_revenue = df["Total_Sales"].sum()
print(f"\nTotal Revenue: ₹{total_revenue:,.2f}")

# Revenue by category
category_sales = df.groupby("Category")["Total_Sales"].sum()
print("\nSales by Category:")
print(category_sales)

# Top products
top_products = df.groupby("Product")["Total_Sales"].sum().sort_values(ascending=False)
print("\nTop Selling Products:")
print(top_products)

# Sales by region
region_sales = df.groupby("Region")["Total_Sales"].sum()
print("\nSales by Region:")
print(region_sales)

# -------------------------------
# 6. Visualization
# -------------------------------

sns.set(style="whitegrid")

# Category-wise sales
plt.figure(figsize=(6,4))
category_sales.plot(kind="bar", title="Sales by Category")
plt.ylabel("Total Sales")
plt.show()

# Region-wise sales
plt.figure(figsize=(6,4))
region_sales.plot(kind="bar", color="green", title="Sales by Region")
plt.ylabel("Total Sales")
plt.show()

# Monthly sales trend
monthly_sales = df.groupby("Month")["Total_Sales"].sum()

plt.figure(figsize=(6,4))
monthly_sales.plot(marker="o", title="Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()
