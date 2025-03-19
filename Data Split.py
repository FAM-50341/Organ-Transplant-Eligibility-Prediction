#Step 6: Split Data into Training & Testing Data

# Define the features (X) and target (y)
X = combined_data[['Age_y', 'Gender_y', 'Blood Group', 'Required Organ', 'Condition_y',
                   'Age_x', 'Gender_x', 'Location', 'Donated Organ', 'Condition_x']]

# Combine both match scores into a final match score column
combined_data['Match Score'] = (combined_data['Match Score_x'] == 1) & (combined_data['Match Score_y'] == 1)
combined_data['Match Score'] = combined_data['Match Score'].astype(int)

# You can now use this single column as the target variable
y = combined_data['Match Score']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

