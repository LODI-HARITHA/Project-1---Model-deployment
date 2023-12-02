import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and train the Naive Bayes model (GaussianNB)
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# Set the title and subtitle of the app
st.title("Naive Bayes Model Deployment")
st.subheader("Predicting Failure Status")


# Add input elements for the variables
V1 = st.number_input("Wind_speed(2m/s to 72m/s)")
V2 = st.number_input("Power(2 MW to 3.5 MW)")
V3 = st.number_input("Nacelle_ambient_temperature(-20°C to 40°C)")
V4 = st.number_input("Generator_bearing_temperature(70  °C to 100°C )")
V5 = st.number_input("Gear_oil_temperature(50 c to 90 c)")
V6 = st.number_input("Ambient_temperature(10°C to 57°C)")
V7 = st.number_input("Rotor_Speed(50-300 RPM)")
V8 = st.number_input("Nacelle_temperature(30°C  to 60°C )")
V9 = st.number_input("Bearing_temperature(70 °C to 100°C)")
V10 = st.number_input("Generator_speed(1000 to 1800 rpm)")
V11= st.number_input("Yaw_angle(0 to 60 Degree)")
V12= st.number_input("Wind_direction(0 to 90 Degree)")
V13 = st.number_input("Wheel_hub_temperature(-40 °C  to 80°C )")
V14 = st.number_input("Gear_box_inlet_temperature(40°C to 60°C)")

# Add more input elements for additional variables as needed
input_data = pd.DataFrame([[V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14]],
                          columns=["Wind_speed", "Power",  "Nacelle_ambient_temperature",
                                   "Generator_bearing_temperature", "Gear_oil_temperature", "Ambient_temperature",
                                   "Rotor_Speed", "Nacelle_temperature", "Bearing_temperature",
                                   "Generator_speed", "Yaw_angle","Wind_direction", "Wheel_hub_temperature",
                                   "Gear_box_inlet_temperature"])

# Adjust the column names and number of columns based on your data

# Make predictions on the input data
prediction = model.predict(input_data)

# Display the prediction
st.write("Prediction:", prediction)










