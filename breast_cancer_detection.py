import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('breast_cancer_dataset.csv')
train_data = pd.read_csv('breast_cancer_dataset_sample.csv')
# tables is created using pandas data frame
doctor_table = pd.DataFrame(columns=['name', 'Did', 'speciality'])
patient_table = pd.DataFrame(columns=['patient_id', 'most_likely_diagnosis', 'date'])
patient_doctor_table = pd.DataFrame(columns=['Pid', 'Did', 'Prescribed_test', 'Test_date'])
breast_scan_test_table = pd.DataFrame(columns=['Test_id', 'Did', 'Pid', 'Test_Type', 'date'])
patient_breast_scan_table = pd.DataFrame(columns=['Pid', 'Did', 'Test_id', 'Breast_attributes', 'Test_Status'])

def standardize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    standardized_X = (X - mean) / std
    return standardized_X

# first Standardize the attributes of the breast dataset and we have to drop diagnosis column
X = data.drop(columns=['diagnosis'])
train = train_data.drop(columns=['diagnosis'])
X_standardized = standardize_data(X)
train_standardized = standardize_data(train)


# use sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    num_samples, num_features = X.shape
    theta = np.zeros(num_features)
    
    for _ in range(num_iterations):
        # Calculate the hypothesis and the gradient
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / num_samples
        
        # Update the parameters (theta)
        theta -= learning_rate * gradient
    
    return theta

# Add bias term to the input features
X_bias = np.c_[np.ones((X_standardized.shape[0], 1)), X_standardized]

# Convert diagnosis labels to binary (0: Benign, 1: Malignant)
y = data['diagnosis'].map({'M': 1, 'B': 0}).values

# Train the logistic regression model
coefficients = logistic_regression(X_bias, y)

def predict(X, coefficients):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    z = np.dot(X_bias, coefficients)
    return sigmoid(z)

# Predict using the training data itself for demonstration
predictions = predict(train_standardized, coefficients)

# Convert the predictions to diagnosis labels
predicted_diagnosis = np.where(predictions >= 0.5, 'M', 'B')
print(predicted_diagnosis)
# Get the absolute values of coefficients
abs_coefficients = np.abs(coefficients[1:])

# Sort the coefficients in descending order and get the corresponding feature names
sorted_indices = np.argsort(abs_coefficients)[::-1]
important_attributes = X.columns[sorted_indices]

#bar chart creation
plt.bar(important_attributes, abs_coefficients[sorted_indices])
plt.xlabel("Attributes")
plt.ylabel("Coefficient Magnitude")
plt.title("Importance of Attributes for Breast Cancer Detection")
plt.xticks(rotation=90)
plt.show()

# Creating plot
# fig = plt.figure(figsize =(10, 7))
# plt.pie(sorted_indices, labels = important_attributes)
 
 
# show plot
# plt.show()

