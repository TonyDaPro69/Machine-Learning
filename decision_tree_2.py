#-------------------------------------------------------------------------
# AUTHOR: Tony Le
# FILENAME: decision_tree_2
# SPECIFICATION: This code implements the decision tree classifier to be tested with different contact lens data sets
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

age_mapping = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
spectacle_mapping = {"Myope": 1, "Hypermetrope": 2}
astigmatism_mapping = {"Yes": 1, "No": 2}
tear_mapping = {"Normal": 1, "Reduced": 2}
class_mapping = {"Yes": 1, "No": 2}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here

    for row in dbTraining:
        X.append([
            age_mapping[row[0]],
            spectacle_mapping[row[1]],
            astigmatism_mapping[row[2]],
            tear_mapping[row[3]]
        ])
        Y.append(class_mapping[row[4]])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    #Loop your training and test tasks 10 times here
    total_accuracy = 0

    for i in range (10):

       #Fitting the decision tree to the data setting max_depth=5
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here

       dbTest = []
       with open("contact_lens_test.csv", 'r') as csvfile:
           reader = csv.reader(csvfile)
           for j, row in enumerate(reader):
               if j > 0:  # Skipping the header
                   dbTest.append(row)

       correct_predictions = 0
       total_predictions = len(dbTest)

       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here

           test_features = [
               age_mapping[data[0]],
               spectacle_mapping[data[1]],
               astigmatism_mapping[data[2]],
               tear_mapping[data[3]]
           ]

           class_predicted = clf.predict([test_features])[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here

           actual_label = class_mapping[data[4]]
           if class_predicted == actual_label:
               correct_predictions += 1

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here

       accuracy = correct_predictions / total_predictions
       total_accuracy += accuracy

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here

    avg_accuracy = total_accuracy / 10
    print(f"Final accuracy when training on {ds}: {avg_accuracy:.2f}")

#Results:
'''
Final accuracy when training on contact_lens_training_1.csv: 0.50
Final accuracy when training on contact_lens_training_2.csv: 0.75
Final accuracy when training on contact_lens_training_3.csv: 0.88
'''

