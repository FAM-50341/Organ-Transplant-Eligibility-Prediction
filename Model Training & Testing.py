#Step 8: Training Models
# Initialize the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='linear')
}

# Ensure all categorical columns are encoded
categorical_features = ['Gender_y', 'Blood Group', 'Required Organ', 'Location', 'Donated Organ', 'Condition_y', 'Condition_x']

# Apply the ColumnTransformer to encode the categorical columns
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Keep the numerical columns as they are
)

# Fit and transform the training data (X_train_data) and transform the test data (X_test_data)
X_train_encoded_data = ct.fit_transform(X_train_data)
X_test_encoded_data = ct.transform(X_test_data)

# Check the transformed data (optional, can help in debugging)
print("Transformed X_train shape:", X_train_encoded_data.shape)
print("Transformed X_test shape:", X_test_encoded_data.shape)

# Train and evaluate models
results = {}
for name, model in models.items():
    try:
        # Fit the model on the transformed training data
        model.fit(X_train_encoded_data, y_train)

        # Make predictions and get probabilities
        y_pred = model.predict(X_test_encoded_data)
        y_prob = model.predict_proba(X_test_encoded_data)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Store results
        results[name] = {"Accuracy": accuracy, "AUC": auc}

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
        plt.title(f"Confusion Matrix for {name}")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC Curve for {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error in model {name}: {e}")

# Display summary of results
results_df = pd.DataFrame(results).T
print("Model Performance Summary:\n", results_df)

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Define model
model = LogisticRegression()

# Check columns in your training data
print(X_train.columns)

# Ensure columns in ColumnTransformer match the ones in the training data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'),
                   ['Age_y', 'Gender_y', 'Blood Group', 'Required Organ', 'Location',
       'Age_x', 'Gender_x', 'Donated Organ', 'Condition_x'])],
    remainder='passthrough'
)
# Simulate some training data (make sure columns match exactly)
train_data = pd.DataFrame({
    'Age_y': [45, 50, 65],
    'Gender_y': ['Male', 'Female', 'Male'],
    'Blood Group': ['A+', 'B+', 'A-'],
    'Required Organ': ['Kidney', 'Heart', 'Kidney'],
    'Location': ['Dhaka', 'Chittagong', 'Dhaka'],
    'Condition_y': ['Critical', 'Stable', 'Critical'],
    'Age_x': [43, 45, 54],
    'Gender_x': ['Female', 'Male', 'Male'],
    'Donated Organ': ['Heart', 'Kidney', 'Kidney'],
    'Condition_x': ['Healthy', 'Healthy', 'Critical']
})

# Example labels
train_labels = ['No Matched', 'Matched', 'No Matched']

# Fit the ColumnTransformer and model on the training data
X_train = train_data.drop('Condition_y', axis=1)
y_train = train_labels
X_encoded = ct.fit_transform(X_train)  # Fit and transform the X data

# Fit the Logistic Regression model
model.fit(X_encoded, y_train)

# User input function with validation
def get_user_input():
    print("Please enter the details for Patient and Donor.")

    # Patient Information
    patient_age = int(input("Enter Patient's Age: "))
    while patient_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        patient_age = int(input("Enter Patient's Age: "))

    patient_gender = input("Enter Patient's Gender (Male/Female): ")
    patient_blood_group = input("Enter Patient's Blood Group (e.g., A, B, AB, O): ")
    patient_required_organ = input("Enter Required Organ (e.g., Kidney, Heart, etc.): ")
    patient_location = input("Enter Patient's Location (e.g., Dhaka, Chittagong, etc.): ")
    patient_condition = input("Enter Patient's Condition (e.g., Critical, Stable, etc.): ")

    # Donor Information
    donor_age = int(input("Enter Donor's Age: "))
    while donor_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        donor_age = int(input("Enter Donor's Age: "))

    donor_gender = input("Enter Donor's Gender (Male/Female): ")
    donor_blood_group = input("Enter Donor's Blood Group (e.g., A, B, AB, O): ")
    donor_donated_organ = input("Enter Donated Organ (e.g., Kidney, Heart, etc.): ")
    donor_location = input("Enter Donor's Location (e.g., Dhaka, Chittagong, etc.): ")
    donor_condition = input("Enter Donor's Condition (e.g., Healthy, Critical, etc.): ")

    patient_data = {
        'Age': patient_age,
        'Gender': patient_gender,
        'Blood Group': patient_blood_group,
        'Required Organ': patient_required_organ,
        'Location': patient_location,
        'Condition': patient_condition
    }

    donor_data = {
        'Age': donor_age,
        'Gender': donor_gender,
        'Blood Group': donor_blood_group,
        'Donated Organ': donor_donated_organ,
        'Location': donor_location,
        'Condition': donor_condition
    }

    return patient_data, donor_data

# Function to predict disease and transplantation eligibility
def predict_disease(patient_data, donor_data):
    """
    patient_data: dict containing the patient's details (Age, Gender, Blood Group, Required Organ, Condition)
    donor_data: dict containing the donor's details (Age, Gender, Blood Group, Donated Organ, Location, Condition)
    """

    # 1. Blood Group, Required Organ, Donated Organ & Location Match Logic:
    if (patient_data['Blood Group'] == donor_data['Blood Group'] and
        patient_data['Required Organ'] == donor_data['Donated Organ'] and
        patient_data['Location'] == donor_data['Location']):
        match_score = 1
    else:
        match_score = 0

    # 2. Prepare the input for prediction
    input_data = {
        'Age_y': [patient_data['Age']],
        'Gender_y': [patient_data['Gender']],
        'Blood Group': [patient_data['Blood Group']],
        'Required Organ': [patient_data['Required Organ']],
        'Location': [patient_data['Location']],
        'Condition_y': [patient_data['Condition']],
        'Age_x': [donor_data['Age']],
        'Gender_x': [donor_data['Gender']],
        'Location': [donor_data['Location']],
        'Donated Organ': [donor_data['Donated Organ']],
        'Condition_x': [donor_data['Condition']]
    }

    input_df = pd.DataFrame(input_data)

    # 3. Encode the input data using the previously fitted column transformer
    input_encoded = ct.transform(input_df)

    # 2. Apply custom rules for determining transplantation eligibility
    # Example rule: If both the patient and donor are in critical condition, eligibility is reduced.
    if patient_data['Condition'] == 'Critical' and donor_data['Condition'] == 'Critical':
        condition_factor = 0.5  # Reduced eligibility for both being in critical condition
    else:
        condition_factor = 1  # Full eligibility

     # Combine these factors (match score, condition factor, age factor) to determine overall eligibility
    posibility = match_score * condition_factor

    # 3. Prepare a message based on eligibility score
    if posibility >= 0.95:
        transplantation_message = "Congratulations! You have been eligible for the transplantation process."
    else:
        transplantation_message = "You are not eligible for transplantation at this time."

    return match_score, posibility , transplantation_message

# Main function to run the whole process
def main():
    patient, donor = get_user_input()

    match_score, posibility , transplantation_message = predict_disease(patient, donor)

    print(f"Match Score: {match_score}")
    print(f"Eligibility: {posibility }")
    print(f"Message: {transplantation_message}")

# Call main function to execute the program
if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Define model
model = RandomForestClassifier()

# Check columns in your training data
print(X_train.columns)

# Ensure columns in ColumnTransformer match the ones in the training data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'),
                   ['Age_y', 'Gender_y', 'Blood Group', 'Required Organ', 'Location',
       'Age_x', 'Gender_x', 'Donated Organ', 'Condition_x'])],
    remainder='passthrough'
)
# Simulate some training data (make sure columns match exactly)
train_data = pd.DataFrame({
    'Age_y': [45, 50, 65],
    'Gender_y': ['Male', 'Female', 'Male'],
    'Blood Group': ['A+', 'B+', 'A-'],
    'Required Organ': ['Kidney', 'Heart', 'Kidney'],
    'Location': ['Dhaka', 'Chittagong', 'Dhaka'],
    'Condition_y': ['Critical', 'Stable', 'Critical'],
    'Age_x': [43, 45, 54],
    'Gender_x': ['Female', 'Male', 'Male'],
    'Donated Organ': ['Heart', 'Kidney', 'Kidney'],
    'Condition_x': ['Healthy', 'Healthy', 'Critical']
})

# Example labels
train_labels = ['No Matched', 'Matched', 'No Matched']

# Fit the ColumnTransformer and model on the training data
X_train = train_data.drop('Condition_y', axis=1)
y_train = train_labels
X_encoded = ct.fit_transform(X_train)  # Fit and transform the X data

# Fit the Logistic Regression model
model.fit(X_encoded, y_train)

# User input function with validation
def get_user_input():
    print("Please enter the details for Patient and Donor.")

    # Patient Information
    patient_age = int(input("Enter Patient's Age: "))
    while patient_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        patient_age = int(input("Enter Patient's Age: "))

    patient_gender = input("Enter Patient's Gender (Male/Female): ")
    patient_blood_group = input("Enter Patient's Blood Group (e.g., A, B, AB, O): ")
    patient_required_organ = input("Enter Required Organ (e.g., Kidney, Heart, etc.): ")
    patient_location = input("Enter Patient's Location (e.g., Dhaka, Chittagong, etc.): ")
    patient_condition = input("Enter Patient's Condition (e.g., Critical, Stable, etc.): ")

    # Donor Information
    donor_age = int(input("Enter Donor's Age: "))
    while donor_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        donor_age = int(input("Enter Donor's Age: "))

    donor_gender = input("Enter Donor's Gender (Male/Female): ")
    donor_blood_group = input("Enter Donor's Blood Group (e.g., A, B, AB, O): ")
    donor_donated_organ = input("Enter Donated Organ (e.g., Kidney, Heart, etc.): ")
    donor_location = input("Enter Donor's Location (e.g., Dhaka, Chittagong, etc.): ")
    donor_condition = input("Enter Donor's Condition (e.g., Healthy, Critical, etc.): ")

    patient_data = {
        'Age': patient_age,
        'Gender': patient_gender,
        'Blood Group': patient_blood_group,
        'Required Organ': patient_required_organ,
        'Location': patient_location,
        'Condition': patient_condition
    }

    donor_data = {
        'Age': donor_age,
        'Gender': donor_gender,
        'Blood Group': donor_blood_group,
        'Donated Organ': donor_donated_organ,
        'Location': donor_location,
        'Condition': donor_condition
    }

    return patient_data, donor_data

# Function to predict disease and transplantation eligibility
def predict_disease(patient_data, donor_data):
    """
    patient_data: dict containing the patient's details (Age, Gender, Blood Group, Required Organ, Condition)
    donor_data: dict containing the donor's details (Age, Gender, Blood Group, Donated Organ, Location, Condition)
    """

    # 1. Blood Group, Required Organ, Donated Organ & Location Match Logic:
    if (patient_data['Blood Group'] == donor_data['Blood Group'] and
        patient_data['Required Organ'] == donor_data['Donated Organ'] and
        patient_data['Location'] == donor_data['Location']):
        match_score = 1
    else:
        match_score = 0

    # 2. Prepare the input for prediction
    input_data = {
        'Age_y': [patient_data['Age']],
        'Gender_y': [patient_data['Gender']],
        'Blood Group': [patient_data['Blood Group']],
        'Required Organ': [patient_data['Required Organ']],
        'Location': [patient_data['Location']],
        'Condition_y': [patient_data['Condition']],
        'Age_x': [donor_data['Age']],
        'Gender_x': [donor_data['Gender']],
        'Location': [donor_data['Location']],
        'Donated Organ': [donor_data['Donated Organ']],
        'Condition_x': [donor_data['Condition']]
    }

    input_df = pd.DataFrame(input_data)

    # 3. Encode the input data using the previously fitted column transformer
    input_encoded = ct.transform(input_df)

    # 2. Apply custom rules for determining transplantation eligibility
    # Example rule: If both the patient and donor are in critical condition, eligibility is reduced.
    if patient_data['Condition'] == 'Critical' and donor_data['Condition'] == 'Critical':
        condition_factor = 0.5  # Reduced eligibility for both being in critical condition
    else:
        condition_factor = 1  # Full eligibility

     # Combine these factors (match score, condition factor, age factor) to determine overall eligibility
    posibility = match_score * condition_factor

    # 3. Prepare a message based on eligibility score
    if posibility >= 0.95:
        transplantation_message = "Congratulations! You have been eligible for the transplantation process."
    else:
        transplantation_message = "You are not eligible for transplantation at this time."

    return match_score, posibility , transplantation_message

# Main function to run the whole process
def main():
    patient, donor = get_user_input()

    match_score, posibility , transplantation_message = predict_disease(patient, donor)

    print(f"Match Score: {match_score}")
    print(f"Eligibility: {posibility }")
    print(f"Message: {transplantation_message}")

# Call main function to execute the program
if __name__ == "__main__":
    main()


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Define model
model = RandomForestClassifier()

# Check columns in your training data
print(X_train.columns)

# Ensure columns in ColumnTransformer match the ones in the training data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'),
                   ['Age_y', 'Gender_y', 'Blood Group', 'Required Organ', 'Location',
       'Age_x', 'Gender_x', 'Donated Organ', 'Condition_x'])],
    remainder='passthrough'
)
# Simulate some training data (make sure columns match exactly)
train_data = pd.DataFrame({
    'Age_y': [45, 50, 65],
    'Gender_y': ['Male', 'Female', 'Male'],
    'Blood Group': ['A+', 'B+', 'A-'],
    'Required Organ': ['Kidney', 'Heart', 'Kidney'],
    'Location': ['Dhaka', 'Chittagong', 'Dhaka'],
    'Condition_y': ['Critical', 'Stable', 'Critical'],
    'Age_x': [43, 45, 54],
    'Gender_x': ['Female', 'Male', 'Male'],
    'Donated Organ': ['Heart', 'Kidney', 'Kidney'],
    'Condition_x': ['Healthy', 'Healthy', 'Critical']
})

# Example labels
train_labels = ['No Matched', 'Matched', 'No Matched']

# Fit the ColumnTransformer and model on the training data
X_train = train_data.drop('Condition_y', axis=1)
y_train = train_labels
X_encoded = ct.fit_transform(X_train)  # Fit and transform the X data

# Fit the Logistic Regression model
model.fit(X_encoded, y_train)

# User input function with validation
def get_user_input():
    print("Please enter the details for Patient and Donor.")

    # Patient Information
    patient_age = int(input("Enter Patient's Age: "))
    while patient_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        patient_age = int(input("Enter Patient's Age: "))

    patient_gender = input("Enter Patient's Gender (Male/Female): ")
    patient_blood_group = input("Enter Patient's Blood Group (e.g., A, B, AB, O): ")
    patient_required_organ = input("Enter Required Organ (e.g., Kidney, Heart, etc.): ")
    patient_location = input("Enter Patient's Location (e.g., Dhaka, Chittagong, etc.): ")
    patient_condition = input("Enter Patient's Condition (e.g., Critical, Stable, etc.): ")

    # Donor Information
    donor_age = int(input("Enter Donor's Age: "))
    while donor_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        donor_age = int(input("Enter Donor's Age: "))

    donor_gender = input("Enter Donor's Gender (Male/Female): ")
    donor_blood_group = input("Enter Donor's Blood Group (e.g., A, B, AB, O): ")
    donor_donated_organ = input("Enter Donated Organ (e.g., Kidney, Heart, etc.): ")
    donor_location = input("Enter Donor's Location (e.g., Dhaka, Chittagong, etc.): ")
    donor_condition = input("Enter Donor's Condition (e.g., Healthy, Critical, etc.): ")

    patient_data = {
        'Age': patient_age,
        'Gender': patient_gender,
        'Blood Group': patient_blood_group,
        'Required Organ': patient_required_organ,
        'Location': patient_location,
        'Condition': patient_condition
    }

    donor_data = {
        'Age': donor_age,
        'Gender': donor_gender,
        'Blood Group': donor_blood_group,
        'Donated Organ': donor_donated_organ,
        'Location': donor_location,
        'Condition': donor_condition
    }

    return patient_data, donor_data

# Function to predict disease and transplantation eligibility
def predict_disease(patient_data, donor_data):
    """
    patient_data: dict containing the patient's details (Age, Gender, Blood Group, Required Organ, Condition)
    donor_data: dict containing the donor's details (Age, Gender, Blood Group, Donated Organ, Location, Condition)
    """

    # 1. Blood Group, Required Organ, Donated Organ & Location Match Logic:
    if (patient_data['Blood Group'] == donor_data['Blood Group'] and
        patient_data['Required Organ'] == donor_data['Donated Organ'] and
        patient_data['Location'] == donor_data['Location']):
        match_score = 1
    else:
        match_score = 0

    # 2. Prepare the input for prediction
    input_data = {
        'Age_y': [patient_data['Age']],
        'Gender_y': [patient_data['Gender']],
        'Blood Group': [patient_data['Blood Group']],
        'Required Organ': [patient_data['Required Organ']],
        'Location': [patient_data['Location']],
        'Condition_y': [patient_data['Condition']],
        'Age_x': [donor_data['Age']],
        'Gender_x': [donor_data['Gender']],
        'Location': [donor_data['Location']],
        'Donated Organ': [donor_data['Donated Organ']],
        'Condition_x': [donor_data['Condition']]
    }

    input_df = pd.DataFrame(input_data)

    # 3. Encode the input data using the previously fitted column transformer
    input_encoded = ct.transform(input_df)

    # 2. Apply custom rules for determining transplantation eligibility
    # Example rule: If both the patient and donor are in critical condition, eligibility is reduced.
    if patient_data['Condition'] == 'Critical' and donor_data['Condition'] == 'Critical':
        condition_factor = 0.5  # Reduced eligibility for both being in critical condition
    else:
        condition_factor = 1  # Full eligibility

     # Combine these factors (match score, condition factor, age factor) to determine overall eligibility
    posibility = match_score * condition_factor

    # 3. Prepare a message based on eligibility score
    if posibility >= 0.95:
        transplantation_message = "Congratulations! You have been eligible for the transplantation process."
    else:
        transplantation_message = "You are not eligible for transplantation at this time."

    return match_score, posibility , transplantation_message

# Main function to run the whole process
def main():
    patient, donor = get_user_input()

    match_score, posibility , transplantation_message = predict_disease(patient, donor)

    print(f"Match Score: {match_score}")
    print(f"Eligibility: {posibility }")
    print(f"Message: {transplantation_message}")

# Call main function to execute the program
if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Define model
model = KNeighborsClassifier()

# Check columns in your training data
print(X_train.columns)

# Ensure columns in ColumnTransformer match the ones in the training data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'),
                   ['Age_y', 'Gender_y', 'Blood Group', 'Required Organ', 'Location',
       'Age_x', 'Gender_x', 'Donated Organ', 'Condition_x'])],
    remainder='passthrough'
)
# Simulate some training data (make sure columns match exactly)
train_data = pd.DataFrame({
    'Age_y': [45, 50, 65],
    'Gender_y': ['Male', 'Female', 'Male'],
    'Blood Group': ['A+', 'B+', 'A-'],
    'Required Organ': ['Kidney', 'Heart', 'Kidney'],
    'Location': ['Dhaka', 'Chittagong', 'Dhaka'],
    'Condition_y': ['Critical', 'Stable', 'Critical'],
    'Age_x': [43, 45, 54],
    'Gender_x': ['Female', 'Male', 'Male'],
    'Donated Organ': ['Heart', 'Kidney', 'Kidney'],
    'Condition_x': ['Healthy', 'Healthy', 'Critical']
})

# Example labels
train_labels = ['No Matched', 'Matched', 'No Matched']

# Fit the ColumnTransformer and model on the training data
X_train = train_data.drop('Condition_y', axis=1)
y_train = train_labels
X_encoded = ct.fit_transform(X_train)  # Fit and transform the X data

# Fit the Logistic Regression model
model.fit(X_encoded, y_train)

# User input function with validation
def get_user_input():
    print("Please enter the details for Patient and Donor.")

    # Patient Information
    patient_age = int(input("Enter Patient's Age: "))
    while patient_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        patient_age = int(input("Enter Patient's Age: "))

    patient_gender = input("Enter Patient's Gender (Male/Female): ")
    patient_blood_group = input("Enter Patient's Blood Group (e.g., A, B, AB, O): ")
    patient_required_organ = input("Enter Required Organ (e.g., Kidney, Heart, etc.): ")
    patient_location = input("Enter Patient's Location (e.g., Dhaka, Chittagong, etc.): ")
    patient_condition = input("Enter Patient's Condition (e.g., Critical, Stable, etc.): ")

    # Donor Information
    donor_age = int(input("Enter Donor's Age: "))
    while donor_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        donor_age = int(input("Enter Donor's Age: "))

    donor_gender = input("Enter Donor's Gender (Male/Female): ")
    donor_blood_group = input("Enter Donor's Blood Group (e.g., A, B, AB, O): ")
    donor_donated_organ = input("Enter Donated Organ (e.g., Kidney, Heart, etc.): ")
    donor_location = input("Enter Donor's Location (e.g., Dhaka, Chittagong, etc.): ")
    donor_condition = input("Enter Donor's Condition (e.g., Healthy, Critical, etc.): ")

    patient_data = {
        'Age': patient_age,
        'Gender': patient_gender,
        'Blood Group': patient_blood_group,
        'Required Organ': patient_required_organ,
        'Location': patient_location,
        'Condition': patient_condition
    }

    donor_data = {
        'Age': donor_age,
        'Gender': donor_gender,
        'Blood Group': donor_blood_group,
        'Donated Organ': donor_donated_organ,
        'Location': donor_location,
        'Condition': donor_condition
    }

    return patient_data, donor_data

# Function to predict disease and transplantation eligibility
def predict_disease(patient_data, donor_data):
    """
    patient_data: dict containing the patient's details (Age, Gender, Blood Group, Required Organ, Condition)
    donor_data: dict containing the donor's details (Age, Gender, Blood Group, Donated Organ, Location, Condition)
    """

    # 1. Blood Group, Required Organ, Donated Organ & Location Match Logic:
    if (patient_data['Blood Group'] == donor_data['Blood Group'] and
        patient_data['Required Organ'] == donor_data['Donated Organ'] and
        patient_data['Location'] == donor_data['Location']):
        match_score = 1
    else:
        match_score = 0

    # 2. Prepare the input for prediction
    input_data = {
        'Age_y': [patient_data['Age']],
        'Gender_y': [patient_data['Gender']],
        'Blood Group': [patient_data['Blood Group']],
        'Required Organ': [patient_data['Required Organ']],
        'Location': [patient_data['Location']],
        'Condition_y': [patient_data['Condition']],
        'Age_x': [donor_data['Age']],
        'Gender_x': [donor_data['Gender']],
        'Location': [donor_data['Location']],
        'Donated Organ': [donor_data['Donated Organ']],
        'Condition_x': [donor_data['Condition']]
    }

    input_df = pd.DataFrame(input_data)

    # 3. Encode the input data using the previously fitted column transformer
    input_encoded = ct.transform(input_df)

    # 2. Apply custom rules for determining transplantation eligibility
    # Example rule: If both the patient and donor are in critical condition, eligibility is reduced.
    if patient_data['Condition'] == 'Critical' and donor_data['Condition'] == 'Critical':
        condition_factor = 0.5  # Reduced eligibility for both being in critical condition
    else:
        condition_factor = 1  # Full eligibility

     # Combine these factors (match score, condition factor, age factor) to determine overall eligibility
    posibility = match_score * condition_factor

    # 3. Prepare a message based on eligibility score
    if posibility >= 0.95:
        transplantation_message = "Congratulations! You have been eligible for the transplantation process."
    else:
        transplantation_message = "You are not eligible for transplantation at this time."

    return match_score, posibility , transplantation_message

# Main function to run the whole process
def main():
    patient, donor = get_user_input()

    match_score, posibility , transplantation_message = predict_disease(patient, donor)

    print(f"Match Score: {match_score}")
    print(f"Eligibility: {posibility }")
    print(f"Message: {transplantation_message}")

# Call main function to execute the program
if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Define model
model = SVC(kernel='linear')

# Check columns in your training data
print(X_train.columns)

# Ensure columns in ColumnTransformer match the ones in the training data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'),
                   ['Age_y', 'Gender_y', 'Blood Group', 'Required Organ', 'Location',
       'Age_x', 'Gender_x', 'Donated Organ', 'Condition_x'])],
    remainder='passthrough'
)
# Simulate some training data (make sure columns match exactly)
train_data = pd.DataFrame({
    'Age_y': [45, 50, 65],
    'Gender_y': ['Male', 'Female', 'Male'],
    'Blood Group': ['A+', 'B+', 'A-'],
    'Required Organ': ['Kidney', 'Heart', 'Kidney'],
    'Location': ['Dhaka', 'Chittagong', 'Dhaka'],
    'Condition_y': ['Critical', 'Stable', 'Critical'],
    'Age_x': [43, 45, 54],
    'Gender_x': ['Female', 'Male', 'Male'],
    'Donated Organ': ['Heart', 'Kidney', 'Kidney'],
    'Condition_x': ['Healthy', 'Healthy', 'Critical']
})

# Example labels
train_labels = ['No Matched', 'Matched', 'No Matched']

# Fit the ColumnTransformer and model on the training data
X_train = train_data.drop('Condition_y', axis=1)
y_train = train_labels
X_encoded = ct.fit_transform(X_train)  # Fit and transform the X data

# Fit the Logistic Regression model
model.fit(X_encoded, y_train)

# User input function with validation
def get_user_input():
    print("Please enter the details for Patient and Donor.")

    # Patient Information
    patient_age = int(input("Enter Patient's Age: "))
    while patient_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        patient_age = int(input("Enter Patient's Age: "))

    patient_gender = input("Enter Patient's Gender (Male/Female): ")
    patient_blood_group = input("Enter Patient's Blood Group (e.g., A, B, AB, O): ")
    patient_required_organ = input("Enter Required Organ (e.g., Kidney, Heart, etc.): ")
    patient_location = input("Enter Patient's Location (e.g., Dhaka, Chittagong, etc.): ")
    patient_condition = input("Enter Patient's Condition (e.g., Critical, Stable, etc.): ")

    # Donor Information
    donor_age = int(input("Enter Donor's Age: "))
    while donor_age > 100:
        print("Age cannot be more than 100. Please enter a valid age.")
        donor_age = int(input("Enter Donor's Age: "))

    donor_gender = input("Enter Donor's Gender (Male/Female): ")
    donor_blood_group = input("Enter Donor's Blood Group (e.g., A, B, AB, O): ")
    donor_donated_organ = input("Enter Donated Organ (e.g., Kidney, Heart, etc.): ")
    donor_location = input("Enter Donor's Location (e.g., Dhaka, Chittagong, etc.): ")
    donor_condition = input("Enter Donor's Condition (e.g., Healthy, Critical, etc.): ")

    patient_data = {
        'Age': patient_age,
        'Gender': patient_gender,
        'Blood Group': patient_blood_group,
        'Required Organ': patient_required_organ,
        'Location': patient_location,
        'Condition': patient_condition
    }

    donor_data = {
        'Age': donor_age,
        'Gender': donor_gender,
        'Blood Group': donor_blood_group,
        'Donated Organ': donor_donated_organ,
        'Location': donor_location,
        'Condition': donor_condition
    }

    return patient_data, donor_data

# Function to predict disease and transplantation eligibility
def predict_disease(patient_data, donor_data):
    """
    patient_data: dict containing the patient's details (Age, Gender, Blood Group, Required Organ, Condition)
    donor_data: dict containing the donor's details (Age, Gender, Blood Group, Donated Organ, Location, Condition)
    """

    # 1. Blood Group, Required Organ, Donated Organ & Location Match Logic:
    if (patient_data['Blood Group'] == donor_data['Blood Group'] and
        patient_data['Required Organ'] == donor_data['Donated Organ'] and
        patient_data['Location'] == donor_data['Location']):
        match_score = 1
    else:
        match_score = 0

    # 2. Prepare the input for prediction
    input_data = {
        'Age_y': [patient_data['Age']],
        'Gender_y': [patient_data['Gender']],
        'Blood Group': [patient_data['Blood Group']],
        'Required Organ': [patient_data['Required Organ']],
        'Location': [patient_data['Location']],
        'Condition_y': [patient_data['Condition']],
        'Age_x': [donor_data['Age']],
        'Gender_x': [donor_data['Gender']],
        'Location': [donor_data['Location']],
        'Donated Organ': [donor_data['Donated Organ']],
        'Condition_x': [donor_data['Condition']]
    }

    input_df = pd.DataFrame(input_data)

    # 3. Encode the input data using the previously fitted column transformer
    input_encoded = ct.transform(input_df)

    # 2. Apply custom rules for determining transplantation eligibility
    # Example rule: If both the patient and donor are in critical condition, eligibility is reduced.
    if patient_data['Condition'] == 'Critical' and donor_data['Condition'] == 'Critical':
        condition_factor = 0.5  # Reduced eligibility for both being in critical condition
    else:
        condition_factor = 1  # Full eligibility

     # Combine these factors (match score, condition factor, age factor) to determine overall eligibility
    posibility = match_score * condition_factor

    # 3. Prepare a message based on eligibility score
    if posibility >= 0.95:
        transplantation_message = "Congratulations! You have been eligible for the transplantation process."
    else:
        transplantation_message = "You are not eligible for transplantation at this time."

    return match_score, posibility , transplantation_message

# Main function to run the whole process
def main():
    patient, donor = get_user_input()

    match_score, posibility , transplantation_message = predict_disease(patient, donor)

    print(f"Match Score: {match_score}")
    print(f"Eligibility: {posibility }")
    print(f"Message: {transplantation_message}")


# Call main function to execute the program
if __name__ == "__main__":
    main()

