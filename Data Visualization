#Step 4: Data Visualization
#Confusion Matrix

# Replace these lists with your actual and predicted values
actual = [1, 0, 1, 0, 1]  # Example actual labels (1 for match, 0 for no match)
predicted = [1, 0, 0, 0, 1]  # Example predicted match labels

# Create the confusion matrix
cm = confusion_matrix(actual, predicted)

# Visualize the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Match', 'Match'], yticklabels=['No Match', 'Match'])
plt.title('Confusion Matrix for Match Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Calculate the correlation matrix

# Select only numeric columns
numeric_columns = patient_data.select_dtypes(include=['number'])

# Calculate the correlation matrix on the numeric columns
correlation_matrix = numeric_columns.corr()
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

#Bloxplot for patient
plt.figure(figsize=(8, 6))
sns.boxplot(x='Blood Group', y='Age', data=patient_data, palette='Set2')
plt.title('Boxplot of Age by Blood Group (Patients)')
plt.xlabel('Blood Group')
plt.ylabel('Age')
plt.show()

#Bloxplot for donor
plt.figure(figsize=(8, 6))
sns.boxplot(x='Blood Group', y='Age', data=patient_data, palette='Set2')
plt.title('Boxplot of Age by Blood Group (Donors)')
plt.xlabel('Blood Group')
plt.ylabel('Age')
plt.show()

# Pairplot for numeric columns
sns.pairplot(patient_data[['Age', 'Match Score']])
plt.title('Pairplot of Numeric Variables')
plt.show()

# Blood Group Distribution of Patient
plt.figure(figsize=(10, 6))
sns.countplot(x='Blood Group', data=patient_data, palette='Set2')
plt.title('Blood Group Distribution (Patients)')
plt.xlabel('Blood Group')
plt.ylabel('Count')
plt.show()

# Blood Group Distribution of Donors
plt.figure(figsize=(10, 6))
sns.countplot(x='Blood Group', data=donor_data, palette='Set2')
plt.title('Blood Group Distribution (Donors)')
plt.xlabel('Blood Group')
plt.ylabel('Count')
plt.show()

# Location Distribution of patient
plt.figure(figsize=(10, 6))
sns.countplot(x='Location', data=patient_data, palette='Set3')
plt.title('Location Distribution (Patients)')
plt.xlabel('Location')
plt.ylabel('Count')
plt.show()

# Location Distribution of train
plt.figure(figsize=(10, 6))
sns.countplot(x='Location', data=donor_data, palette='Set3')
plt.title('Location Distribution (Donors)')
plt.xlabel('Location')
plt.ylabel('Count')
plt.show()

