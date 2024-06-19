# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the training and testing datasets
train_df = pd.read_csv('input_data/titanic/train.csv')
test_df = pd.read_csv('input_data/titanic/test.csv')

# Display the first few rows of the datasets
print("First few rows of the training dataset:")
print(train_df.head())

print("\nFirst few rows of the testing dataset:")
print(test_df.head())

# Data Preparation for training data
# Handle missing values
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)

# Drop columns that are not needed
train_df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# Split the training data into features and target variable
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Preparation for testing data
# Handle missing values
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Drop columns that are not needed
test_df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Encode categorical variables
test_df['Sex'] = label_encoder.fit_transform(test_df['Sex'])
test_df['Embarked'] = label_encoder.fit_transform(test_df['Embarked'])

# Split the testing data into features
X_test = test_df

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Model Building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the model on the validation set
accuracy = accuracy_score(y_val, y_val_pred)
conf_matrix = confusion_matrix(y_val, y_val_pred)
class_report = classification_report(y_val, y_val_pred)

print("\nValidation Accuracy:", accuracy)
print("\nValidation Confusion Matrix:\n", conf_matrix)
print("\nValidation Classification Report:\n", class_report)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Save the predictions to a CSV file
predictions = pd.DataFrame({'PassengerId': test_df.index + 1, 'Survived': y_test_pred})
predictions.to_csv('output_data/titanic/titanic_predictions.csv', index=False)

print("Predictions saved to 'titanic_predictions.csv'.")
