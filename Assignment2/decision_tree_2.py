#-------------------------------------------------------------------------
# AUTHOR: Minh Nhat Doan
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and test a decision tree on the contact lens dataset using three different training sets. 
#              The tree is built with max_depth=5, and the model is evaluated 10 times on the same test set.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-------------------------------------------------------------------------

from sklearn import tree
import csv

# Mapping dictionaries for features and class labels
age_map = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
prescription_map = {"Myope": 1, "Hypermetrope": 2}
astigmatism_map = {"Yes": 1, "No": 2}
tear_map = {"Normal": 1, "Reduced": 2}
# For the output class: "Yes" -> 1, "No" -> 2
class_map = {"Yes": 1, "No": 2}

# List of training set filenames
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # Transform the original categorical training features to numbers and add to the 4D array X.
    # For example: Young = 1, Prepresbyopic = 2, Presbyopic = 3,...
    for row in dbTraining:
        # Assuming the columns are in the order: Age, Spectacle Prescription, Astigmatism, Tear Production Rate, Class
        instance = [
            age_map[row[0]],
            prescription_map[row[1]],
            astigmatism_map[row[2]],
            tear_map[row[3]]
        ]
        X.append(instance)

    # Transform the original categorical training classes to numbers and add to the vector Y.
    # For example: Yes = 1, No = 2.
    for row in dbTraining:
        Y.append(class_map[row[4]])

    total_accuracy = 0.0  # to sum accuracies over the 10 runs

    # Loop the training and testing tasks 10 times
    for i in range(10):

        # Fitting the decision tree to the data with max_depth=5
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        # Read the test data and add to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0:  # skip header
                    dbTest.append(row)

        correct = 0
        total = 0

        for data in dbTest:
            # Transform the features of the test instance to numbers using the same approach as for training.
            test_instance = [
                age_map[data[0]],
                prescription_map[data[1]],
                astigmatism_map[data[2]],
                tear_map[data[3]]
            ]
            # Use the decision tree to make the class prediction. [0] extracts the integer prediction.
            class_predicted = clf.predict([test_instance])[0]

            # Transform the true class label from the test instance.
            true_label = class_map[data[4]]

            # Compare the prediction with the true label to calculate accuracy.
            if class_predicted == true_label:
                correct += 1
            total += 1

        accuracy = correct / total
        total_accuracy += accuracy

    # Find the average accuracy over the 10 runs
    average_accuracy = total_accuracy / 10

    # Print the average accuracy for this training set.
    print("final accuracy when training on {}: {}".format(ds, average_accuracy))
