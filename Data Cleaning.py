#Step 3: Data Cleaning
# Check for missing values in both datasets
print(patient_data.isnull().sum())
print(donor_data.isnull().sum())

# Option 1: Drop rows with missing values
patient_data = patient_data.dropna()
donor_data = donor_data.dropna()

# Remove duplicates from both datasets
patient_data = patient_data.drop_duplicates()
donor_data = donor_data.drop_duplicates()

# Check again for duplicates
print(f"Patient duplicates: {patient_data.duplicated().sum()}")
print(f"Donor duplicates: {donor_data.duplicated().sum()}")

# Convert 'blood_group' and 'location' to categorical types (if they're not already)
patient_data['Blood Group'] = patient_data['Blood Group'].astype('category')
donor_data['Blood Group'] = donor_data['Blood Group'].astype('category')

patient_data['Location'] = patient_data['Location'].astype('category')
donor_data['Location'] = donor_data['Location'].astype('category')

# Check the data types of the columns
print(patient_data.dtypes)
print(donor_data.dtypes)

# Standardize 'Blood Group' and 'Location' to lowercase
patient_data['Blood Group'] = patient_data['Blood Group'].str.lower()
donor_data['Blood Group'] = donor_data['Blood Group'].str.lower()

patient_data['Location'] = patient_data['Location'].str.lower()
donor_data['Location'] = donor_data['Location'].str.lower()

# Handle any other inconsistency (e.g., replacing "unknown" with NaN or similar)
patient_data['Location'] = patient_data['Location'].replace('unknown', 'not available')
donor_data['Location'] = donor_data['Location'].replace('unknown', 'not available')

# Remove outliers (this depends on your analysis and dataset)
patient_data = patient_data[patient_data['Age'] < 100]
donor_data = donor_data[donor_data['Age'] < 100]

# Check the cleaned data
print(patient_data.head())
print(donor_data.head())

# Save cleaned datasets
patient_data.to_csv('cleaned_patient_dataset.csv', index=False)
donor_data.to_csv('cleaned_donor_dataset.csv', index=False)
