import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Creating a mock dataset
data = {
    'Study_Hours': [2, 5, 1, 9, 7, 3, 8, 2, 10, 5],
    'Attendance_Percentage': [70, 85, 60, 95, 80, 75, 90, 65, 98, 82],
    'Previous_Score': [50, 70, 45, 90, 85, 55, 88, 52, 95, 72],
    'Final_Grade': [55, 75, 48, 92, 88, 60, 91, 56, 97, 78]
}

df = pd.DataFrame(data)
print("--- Sample Data ---")
print(df.head())
# Features (X) and Target (y)
X = df[['Study_Hours', 'Attendance_Percentage', 'Previous_Score']]
y = df['Final_Grade']

# Splitting the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Complete.")
# Function to predict performance based on user input
def predict_student_grade(hours, attendance, prev_score):
    input_data = np.array([[hours, attendance, prev_score]])
    prediction = model.predict(input_data)
    return prediction[0]

# Example Prediction
sample_hours = 6
sample_attendance = 88
sample_prev_score = 75

predicted_grade = predict_student_grade(sample_hours, sample_attendance, sample_prev_score)

print(f"\n--- Prediction Result ---")
print(f"Input: {sample_hours}hrs study, {sample_attendance}% attendance, {sample_prev_score} prev score")
print(f"Predicted Final Grade: {predicted_grade:.2f}")

# Evaluating the model on the test set
y_pred = model.predict(X_test)
print(f"\nModel Accuracy (R2 Score): {r2_score(y_test, y_pred):.4f}")