#-------------------------------------------------------------------------
# AUTHOR: Tony Le
# FILENAME: knn
# SPECIFICATION: This program computes the LOO-CV error rate for a 1NN classifier
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Loop your data to allow each instance to be your test set
label_mapping = {"ham": 0, "spam": 1}
total_instances = len(db)
error_count = 0

for i in db:
    X = []
    Y = []
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    for train_instance in db:
        if train_instance != i:
            X.append([float(num) for num in train_instance[:-1]])

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

            Y.append(label_mapping[train_instance[-1]])

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here

    testSample = [float(num) for num in i[:-1]]
    actual_label = label_mapping[i[-1]]

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here

    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

    if class_predicted != actual_label:
        error_count += 1

#Print the error rate
#--> add your Python code here

error_rate = error_count / total_instances
print(f"Final LOO-CV error rate: {error_rate:.2f}")

#Result:
#Final LOO-CV error rate: 0.14




