#-------------------------------------------------------------------------
# AUTHOR: Minh Nhat Doan
# FILENAME: decision_tree.py
# SPECIFICATION: This program reads the 'contact_lens.csv' file, transforms
#                the categorical features and labels into numbers, fits a
#                decision tree classifier using entropy, and then plots the tree.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 8 hours
#-------------------------------------------------------------------------

# Importing required libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

# Initialize lists for the database and features/labels
db = []
X = []
Y = []

# Reading the data from the csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skipping the header
            db.append(row)
            print(row)

# Create mapping dictionaries for each categorical attribute
age_mapping = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
spectacle_mapping = {"Myope": 1, "Hypermetrope": 2}
astigmatism_mapping = {"Yes": 1, "No": 2}
tear_mapping = {"Reduced": 1, "Normal": 2}

# Create mapping for the class labels (Recommended Lenses)
class_mapping = {"Yes": 1, "No": 2}

# Transform the categorical features into numbers and add them to X
for row in db:
    temp = []
    temp.append(age_mapping[row[0]])
    temp.append(spectacle_mapping[row[1]])
    temp.append(astigmatism_mapping[row[2]])
    temp.append(tear_mapping[row[3]])
    X.append(temp)

# Transform the class labels into numbers and add them to Y
for row in db:
    Y.append(class_mapping[row[4]])

# Fitting the decision tree to the data using entropy
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

# Plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes', 'No'], filled=True, rounded=True)
plt.show()
