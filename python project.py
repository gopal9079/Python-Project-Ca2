import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

# ----------------------------
# LOAD DATASET
# ----------------------------
file_path = r'C:\Users\dell\Desktop\GopalPython\34_Details_Of_Assembly_Segment_Of_PC.csv'
print("File exists:", os.path.exists(file_path))

# Load the dataset
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
    exit()

# ----------------------------
# DATA CLEANING
# ----------------------------
print("Missing values per column:\n", df.isnull().sum())

# Handle missing values
df['Votes Secured Evm'] = pd.to_numeric(df['Votes Secured Evm'], errors='coerce')
df.dropna(subset=['Votes Secured Evm', 'Candidate Name', 'Party'], inplace=True)

# Ensure correct data types
df['Pc No'] = df['Pc No'].astype(int)

# ----------------------------
# DATA ANALYSIS
# ----------------------------

# Number of votes per party
party_votes = df.groupby('Party')['Votes Secured Evm'].sum()
print("\nTotal Votes by Party:\n", party_votes)

# Most common parliamentary constituencies
top_constituencies = df['PC Name'].value_counts().head(10)
print("\nMost Common Parliamentary Constituencies:\n", top_constituencies)

# Top 10 candidates by votes
top_candidates = df[['Candidate Name', 'Votes Secured Evm']].sort_values(by='Votes Secured Evm', ascending=False).head(10)
print("\nTop Candidates by Votes:\n", top_candidates)

# ----------------------------
# DATA VISUALIZATION
# ----------------------------

# Bar chart: Top 10 parties by votes
top_parties = party_votes.sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_parties.values, y=top_parties.index, hue=top_parties.index, palette='viridis', legend=False)
plt.title('Top 10 Parties by Total Votes')
plt.xlabel('Total Votes')
plt.ylabel('Party')
plt.tight_layout()
plt.show()

# Pie chart: Party vote share (top 5 + others)
top5_parties = party_votes.sort_values(ascending=False).head(5)
others = party_votes.sum() - top5_parties.sum()
pie_data = pd.concat([top5_parties, pd.Series({'Others': others})])

plt.figure(figsize=(8, 8))
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
plt.title('Vote Share of Top 5 Parties')
plt.show()

# Histogram: Distribution of votes
plt.figure(figsize=(10, 6))
sns.histplot(df['Votes Secured Evm'], bins=50, kde=True)
plt.title('Distribution of Votes Secured per Candidate')
plt.xlabel('Votes')
plt.ylabel('Frequency')
plt.xlim(0, df['Votes Secured Evm'].quantile(0.95))  # Focus on lower 95%
plt.tight_layout()
plt.show()

# ----------------------------
# FILTERING & QUERYING
# ----------------------------

# Example 1: Filter high vote candidates from Andhra Pradesh
ap_high_votes = df[(df['State/Ut Name'] == 'Andhra Pradesh') & (df['Votes Secured Evm'] > 50000)]
print("\nAndhra Pradesh Candidates with Votes > 50,000:\n", ap_high_votes[['Candidate Name', 'Party', 'Votes Secured Evm']])

# Example 2: Query results for a specific constituency
constituency = 'Araku'
constituency_results = df[df['PC Name'] == constituency]
print(f"\nResults for Constituency '{constituency}':\n", constituency_results[['Candidate Name', 'Party', 'Votes Secured Evm']])

# ----------------------------
# MACHINE LEARNING: PREDICTION
# ----------------------------

# Prepare data for regression (predicting votes based on total electors in PC)
ml_data = df[['Total Electors In PC', 'Votes Secured Evm']].dropna()
X = ml_data[['Total Electors In PC']]
y = ml_data['Votes Secured Evm']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel Performance:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}")

# Predict future votes based on hypothetical total electors
future_electors = pd.DataFrame({'Total Electors In PC': [100000, 200000, 300000]})
future_predictions = model.predict(future_electors)
print("\nPredicted Votes for Future Electors:\n", future_electors.assign(Predicted_Votes=future_predictions))

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(future_electors['Total Electors In PC'], future_predictions, marker='o', label='Predicted Votes')
plt.title('Predicted Votes by Total Electors')
plt.xlabel('Total Electors in PC')
plt.ylabel('Predicted Votes')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
