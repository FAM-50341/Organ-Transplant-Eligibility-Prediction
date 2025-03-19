# Step 7: Encoding attributes
# Assuming X is your original DataFrame containing the necessary columns
# Ensure X_train contains all the relevant columns
X_train_data = pd.DataFrame(X_train, columns=['Age_y', 'Gender_y', 'Blood Group', 'Required Organ', 'Condition_y',
                   'Age_x', 'Gender_x', 'Location', 'Donated Organ', 'Condition_x'])

# Define the categorical features
categorical_features = ['Gender_y', 'Blood Group', 'Required Organ', 'Location', 'Donated Organ']

# Create a column transformer to apply OneHotEncoder to the categorical columns
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Keep the numerical columns as they are
)

# Fit the column transformer to X_train and transform it
X_train_encoded = ct.fit_transform(X_train_data)

# Get the new column names from the one-hot encoding process
encoded_columns = ct.transformers_[0][1].get_feature_names_out(categorical_features)

# Combine the new column names with the remaining columns
new_column_names = list(encoded_columns) + [col for col in X_train_data.columns if col not in categorical_features]

# Convert the transformed data into a DataFrame with new column names
X_train_encoded_data = pd.DataFrame(X_train_encoded, columns=new_column_names)

# Check the transformed data
print("Transformed training data shape:", X_train_encoded_data.shape)
print("Column names after encoding:", X_train_encoded_data.columns)

