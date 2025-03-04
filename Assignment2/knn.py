#-------------------------------------------------------------------------
# AUTHOR: Minh Nhat Doan
# FILENAME: knn.py
# SPECIFICATION: Compute the LOO-CV error rate for a 1NN classifier on the spam/ham email classification task.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour 30 minutes
#-------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
import csv

# Read the data from email_classification.csv
db = []
with open('email_classification.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skip the header row
            db.append(row)

errors = 0
n = len(db)

# Loop over each instance for LOO-CV
for idx in range(n):
    X_train = []
    Y_train = []
    
    # Build training set (all instances except the one at index idx)
    for j in range(n):
        if j == idx:
            continue
        row = db[j]
        # Convert the 20 word frequency features to floats
        features = [float(x) for x in row[:-1]]
        X_train.append(features)
        # Map the class label to a number: spam -> 1, ham -> 2
        label_str = row[-1].strip().lower()
        if label_str == "spam":
            Y_train.append(1)
        else:
            Y_train.append(2)
    
    # Prepare the test sample (instance at index idx)
    test_row = db[idx]
    X_test = [float(x) for x in test_row[:-1]]
    true_label = 1 if test_row[-1].strip().lower() == "spam" else 2
    
    # Fit the 1NN classifier
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf.fit(X_train, Y_train)
    
    # Predict the class for the test sample
    prediction = clf.predict([X_test])[0]
    
    # Compare prediction with true label
    if prediction != true_label:
        errors += 1

# Compute the LOO-CV error rate
error_rate = errors / n

# Print the error rate
print("LOO-CV error rate for 1NN: {:.2f}".format(error_rate))
