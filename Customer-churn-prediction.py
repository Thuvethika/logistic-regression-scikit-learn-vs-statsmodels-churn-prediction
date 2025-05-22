
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("customer_churn_dataset.csv")

# Print first 5 lines
print("head")
print(data.head())

#print info
print("info")
print(data.info())

# Check for missing values in the dataset
print(data.isnull().sum())

# Get rows with missing values
missing_data = data[data.isnull().any(axis=1)]

# Display the records with missing values
print(missing_data)

data = data.dropna()
data = data.reset_index(drop=True)

# Display the cleaned dataset
print(data)

#Recheck for missing values in the dataset
print(data.isnull().sum())

# Remove customerid column 
data = data.drop('CustomerID', axis=1)

#print describe
print("describe")
print(data.describe())

# Visualizing the distribution of the target variable
# Code adapted form geeksforgeeks.org, 2020
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=data, hue='Churn', palette='gist_rainbow', legend=False)
plt.title('Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Count the occurrences of Churn count
churn_count = data['Churn'].value_counts()

# Create the pie chart
# Code adapted form Matplotlib 3.7.2 documentation. (n.d.)
plt.figure(figsize=(6, 4))
plt.pie(churn_count, labels= churn_count.index, autopct='%1.1f%%')
plt.title("Customer's Churn Type Distribution")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(data['Age'], bins=30 ,kde=True)
plt.title('Distribution Of Age')
plt.xlabel('Age')
plt.show()

data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 25, 35, 45, 55, 65], labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
data['AgeGroup'] = data['AgeGroup'].astype(object)
data['AgeGroup'].value_counts()

# Set a colorful Seaborn style
sns.set_style("whitegrid")

# Create the figure
plt.figure(figsize=(6,4))

# Create a count plot (bar plot) for AgeGroup
# Waskom, M. (2012). seaborn.countplot — seaborn 0.9.0 documentation.
ax = sns.countplot(x='AgeGroup', data=data, order=data['AgeGroup'].value_counts().index, color='blue')
plt.title('Distribution of Customers by Age Group', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Age Group', fontsize=14, labelpad=10)
plt.ylabel('Number of Customers', fontsize=14, labelpad=10)

# Customize tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add data labels on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

# Remove top and right spines for a cleaner look
sns.despine()

# Show the plot
plt.tight_layout()
plt.show()

#Churn Rate by age
age_churn = data.groupby('Age')['Churn'].mean().reset_index()
plt.figure(figsize=(6,4))
sns.lineplot(x='Age', y='Churn', data=age_churn)
plt.title('Churn Rate by Age')
plt.xlabel('Age')
plt.ylabel('Churn Rate')
plt.show()

#Gender Distribution count plot
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count', )
plt.show()

# Count the occurrences of each gender
gender_counts = data['Gender'].value_counts()

#Gender Distribution pie chart
plt.figure(figsize=(6,4))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.show()

# Group by Gender and calculate churn rate
churn_rate = data.groupby('Gender')['Churn'].mean() * 100  # Convert to percentage
plt.figure(figsize=(6,4))
churn_rate.plot(kind='bar', color=['skyblue', 'pink'])
plt.xlabel('Gender')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Gender')

# Add percentage labels on top of bars
for i, value in enumerate(churn_rate):
    plt.text(i, value + 1, f'{value:.1f}%', ha='center', fontsize=12)

# Display the chart
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Group by 'Tenure' and calculate the mean churn rate
tenure_churn_rate = data.groupby('Tenure')['Churn'].mean().reset_index()
tenure_churn_rate.columns = ['Tenure', 'Churn Rate']

# Visualize the churn rate by tenure
plt.figure(figsize=(10,6))
sns.barplot(x='Tenure', y='Churn Rate', data=tenure_churn_rate)
plt.xlabel("Tenure (Months)")
plt.ylabel("Churn Rate")
plt.title("Churn Rate by Tenure")
plt.show()

# Visualize the churn rate by Usage Frequency
churn_rate = data.groupby('Usage Frequency')['Churn'].mean()

plt.figure(figsize=(10,6))
sns.barplot(x=churn_rate.index, y=churn_rate.values)
plt.xlabel("Usage Frequency Category")
plt.ylabel("Churn Rate")
plt.title("Churn Rate by Usage Frequency Category")
plt.show()

#Subscription Type countplot
plt.figure(figsize=(6,4))
sns.countplot(x='Subscription Type',data=data)
plt.title('Subscribtion Type count plot')
plt.show()

subscription_counts = data['Subscription Type'].value_counts()
plt.figure(figsize=(6,4))
plt.pie(subscription_counts, labels=subscription_counts.index, autopct='%1.1f%%')
plt.title("Customer's Subscription Type Distribution")
plt.show()

# Visualize the churn rate by Subscription Type
churn_rate = data.groupby('Subscription Type')['Churn'].mean()

plt.figure(figsize=(6,4))
sns.barplot(x=churn_rate.index, y=churn_rate.values)
plt.xlabel("Subscription Type")
plt.ylabel("Churn Rate")
plt.title("Churn Rate by Subscription Type")
plt.show()

# Count the occurrences of each Contract Length
Contract_Length = data['Contract Length'].value_counts()

# Create the pie chart
plt.figure(figsize=(6,4))
plt.pie(Contract_Length,labels=Contract_Length.index,autopct='%1.1f%%', colors=['lightblue', 'orange', 'green'])
plt.title("Customer's Contract Length Distribution")
plt.show()

churn_rate = data.groupby('Contract Length')['Churn'].mean()

plt.figure(figsize=(6,4))
sns.barplot(x=churn_rate.index, y=churn_rate.values)
plt.xlabel("Contract Length")
plt.ylabel("Churn Rate")
plt.title("Churn Rate by Contract Length")
plt.show()

#make the Total Spend column into int
data['Total Spend'] = data['Total Spend'].astype(int)

#histogram of Total Spend column
plt.figure(figsize=(6,4))
plt.hist(data['Total Spend'], bins=30)
plt.title('Histogram of Total Spend')
plt.show()

data['Last Interaction'] = data['Last Interaction'].astype(int)

# Group by 'Last Interaction' and calculate the mean churn rate
churn_rate_last_interaction = data.groupby('Last Interaction')['Churn'].mean().reset_index()

# Plotting the churn rate by last interaction
plt.figure(figsize=(6,4))
sns.lineplot(x='Last Interaction', y='Churn', data=churn_rate_last_interaction)
plt.title('Churn Rate by Last Interaction')
plt.xlabel('Days Since Last Interaction')
plt.ylabel('Churn Rate')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Subscription Type', hue='Churn', data=data)
plt.title('Subscription Type vs Churn')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Contract Length',hue='Churn' ,data=data)
plt.title('Contract Length VS Churn')
plt.show()

# Select all numerical columns (float + int) while excluding the 'Churn' column
numerical_features = data.select_dtypes(include=['int', 'float']).drop('Churn', axis=1, errors='ignore')

# Boxplot for each numerical feature
for feature in numerical_features.columns:
    plt.figure(figsize=(4, 4))
    sns.boxplot(x=feature, data=data)
    plt.title(feature)
    plt.show()

# Calculate the average usage frequency by gender
usage_frequency_by_gender = data.groupby('Gender')['Usage Frequency'].mean()

# Display the result
print(usage_frequency_by_gender)

# Convert the Series to a DataFrame
usage_frequency_by_gender_data = usage_frequency_by_gender.reset_index()

# Create a bar plot for Usage Frequency by Gender
plt.figure(figsize=(6,4))
sns.barplot(x='Gender', y='Usage Frequency', data=usage_frequency_by_gender_data)
plt.xlabel('Gender')
plt.ylabel('Usage Frequency')
plt.title('Usage Frequency by Gender')

for i, value in enumerate(usage_frequency_by_gender_data['Usage Frequency']):
    plt.text(i, value + 0.1, f'{value:.2f}', ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# Create a new feature Avg_Spend_per_Month
data['Avg_Spend_per_Month'] = data['Total Spend'] / data['Tenure']

# Display the first few rows to verify the new feature
print(data[['Total Spend', 'Tenure', 'Avg_Spend_per_Month']].head())

# Calculate the churn rate for each Avg_Spend_per_Month
churn_rate_per_spend = data.groupby('Avg_Spend_per_Month')['Churn'].mean().reset_index()

# Create the scatter plot with a regression line
plt.figure(figsize=(12, 6))
sns.regplot(x='Avg_Spend_per_Month', y='Churn', data=churn_rate_per_spend, scatter_kws={'s': 50}, line_kws={'color': 'pink'})
plt.xlabel('Average Spend per Month', fontsize=14)
plt.ylabel('Churn Rate', fontsize=14)
plt.title('Relationship between Average Spend per Month and Churn Rate', fontsize=16)
plt.tight_layout()
plt.show()

# Convert categorical variables
data = pd.get_dummies(data, drop_first=True) 

# Split into features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#code adapted from scikit-learn. (2014). sklearn.linear_model.LogisticRegression 
# Train Logistic Regression model
sklearn_model = LogisticRegression(penalty='l2', solver='newton-cholesky', max_iter=1000)
sklearn_model.fit(x_train, y_train)

# Make predictions
y_pred_sklearn = sklearn_model.predict(x_test)

# Evaluate performance
print("Scikit-learn Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_sklearn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_sklearn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_sklearn))


#code adapted from Guy, D. D. (2022, September 21). Using Logisitic Regression with StatsModel 
# Add constant (intercept) for Statsmodels
x_train_sm = sm.add_constant(x_train)
x_test_sm = sm.add_constant(x_test)

# Train logistic regression model
statsmodels_model = sm.Logit(y_train, x_train_sm)
statsmodels_result = statsmodels_model.fit()

# Summary of the model
print(statsmodels_result.summary())

# Predictions (convert probabilities to class labels: 0 or 1)
y_pred_statsmodels = (statsmodels_result.predict(x_test_sm) >= 0.5).astype(int)

# Evaluate model
print("Statsmodels Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_statsmodels))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_statsmodels))
print("\nClassification Report:\n", classification_report(y_test, y_pred_statsmodels))

# Calculate performance metrics for Scikit-learn
metrics_sklearn = {
    "Accuracy": accuracy_score(y_test, y_pred_sklearn),
    "Precision": precision_score(y_test, y_pred_sklearn),
    "Recall": recall_score(y_test, y_pred_sklearn),
    "F1-Score": f1_score(y_test, y_pred_sklearn)
}

# Calculate performance metrics for Statsmodels
metrics_statsmodels = {
    "Accuracy": accuracy_score(y_test, y_pred_statsmodels),
    "Precision": precision_score(y_test, y_pred_statsmodels),
    "Recall": recall_score(y_test, y_pred_statsmodels),
    "F1-Score": f1_score(y_test, y_pred_statsmodels)
}

# Create a DataFrame to compare metrics
comparison_df = pd.DataFrame([metrics_sklearn, metrics_statsmodels], index=['Scikit-learn', 'Statsmodels'])
print("\nModel Performance Comparison:\n")
print(comparison_df)

# Generate the confusion matrix
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)

# Plot the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Scikit-learn')
plt.show()

# Generate the confusion matrix for Statsmodels
cm_statsmodels = confusion_matrix(y_test, y_pred_statsmodels)

# Plot the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm_statsmodels, annot=True, fmt='d', cmap='Greens', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Statsmodels')
plt.show()

# Create a DataFrame with actual and predicted values
results_sklearn = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_sklearn})

# Calculate the counts of actual and predicted values
actual_counts_sklearn = results_sklearn['Actual'].value_counts().sort_index()
predicted_counts_sklearn = results_sklearn['Predicted'].value_counts().sort_index()

# Create a bar plot
labels = ['No Churn', 'Churn']
x = np.arange(len(labels))  # the label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, actual_counts_sklearn, width, label='Actual')
rects2 = ax.bar(x + width/2, predicted_counts_sklearn, width, label='Predicted')

# Labels, title, and ticks
ax.set_xlabel('Churn')
ax.set_ylabel('Count')
ax.set_title('Actual vs Predicted Churn (Scikit-learn)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)

fig.tight_layout()
plt.show()


# Create a DataFrame with actual and predicted values
results_statsmodels = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_statsmodels})

# Calculate the counts of actual and predicted values
actual_counts_statsmodels = results_statsmodels['Actual'].value_counts().sort_index()
predicted_counts_statsmodels = results_statsmodels['Predicted'].value_counts().sort_index()

# Create a bar plot
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, actual_counts_statsmodels, width, label='Actual', color='skyblue')
rects2 = ax.bar(x + width/2, predicted_counts_statsmodels, width, label='Predicted', color='lightgreen')

# Labels, title, and ticks
ax.set_xlabel('Churn')
ax.set_ylabel('Count')
ax.set_title('Actual vs Predicted Churn (Statsmodels)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

add_labels(rects1)
add_labels(rects2)

fig.tight_layout()
plt.show()
