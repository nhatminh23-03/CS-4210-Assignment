#-------------------------------------------------------------------------
# AUTHOR: Minh Nhat Doan
# FILENAME: naive_bayes.py
# SPECIFICATION: This program reads weather_training.csv to build a Naïve Bayes
#                classifier for predicting whether to play tennis. It then reads
#                weather_test.csv (which contains 10 test instances) and prints
#                the details of each test instance (including the day ID and
#                original features) along with the predicted class and the
#                classification confidence if the confidence is >= 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-------------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB
import csv

# Define mapping dictionaries for converting categorical values to numbers.
# These mappings must be consistent between training and test data.
outlook_map = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature_map = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity_map = {"High": 1, "Normal": 2}
wind_map = {"Weak": 1, "Strong": 2}
# Mapping for class attribute (PlayTennis): Yes -> 1, No -> 2.
play_map = {"Yes": 1, "No": 2}
# Inverse mapping for printing predicted class labels.
inv_play_map = {1: "Yes", 2: "No"}

#--------------------- Training Phase ---------------------
# Read the training data from weather_training.csv.
training_data = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skip header row.
            training_data.append(row)

# Transform the training features (categorical) to numbers and build X and Y.
X = []  # Feature list.
Y = []  # Class label list.
for row in training_data:
    # Convert each feature value to a number.
    # For example: "Sunny" becomes 1, "Mild" becomes 2,...
    features = [
        outlook_map[row[1]],
        temperature_map[row[2]],
        humidity_map[row[3]],
        wind_map[row[4]]
    ]
    X.append(features)
    # Convert the class label (PlayTennis) to a number.
    Y.append(play_map[row[5]])

#Fitting the naive bayes to the data.
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#--------------------- Testing Phase ---------------------
# Read the test data from weather_test.csv.
test_data = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skip header row.
            test_data.append(row)

# Print the header of the solution.
print("{:<8} {:<10} {:<12} {:<10} {:<6} {:<10} {:<10}".format(
    "Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence"
))

# For each test instance, make a prediction if the classification confidence is >= 0.75.
for row in test_data:
    # Extract the original information.
    day = row[0]
    outlook = row[1]
    temperature = row[2]
    humidity = row[3]
    wind = row[4]

    # Convert the test instance's features using the same mappings.
    test_features = [
        outlook_map[outlook],
        temperature_map[temperature],
        humidity_map[humidity],
        wind_map[wind]
    ]

    # Get the predicted probabilities for each class.
    probs = clf.predict_proba([test_features])[0]
    # Get the predicted class (as a number).
    pred_class = clf.predict([test_features])[0]

    # Determine the probability (confidence) of the predicted class.
    # Since clf.classes_ is sorted (should be [1, 2]), if pred_class is 1 then the probability is probs[0], otherwise probs[1].
    if pred_class == 1:
        confidence = probs[0]
    else:
        confidence = probs[1]

    # Convert the predicted class number to a string.
    predicted_class_str = inv_play_map[pred_class]  # "Yes" or "No"

    if confidence >= 0.75:
        print("{:<8} {:<10} {:<12} {:<10} {:<6} {:<10} {:.2f}".format(
            day, outlook, temperature, humidity, wind, predicted_class_str, confidence
        ))
