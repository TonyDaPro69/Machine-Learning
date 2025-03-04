#-------------------------------------------------------------------------
# AUTHOR: Tony Le
# FILENAME: naive_bayes
# SPECIFICATION: This program will output the confidence level of each instance in the file weather_test.csv if confidence is above 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv
dbTraining = []

#Reading the training data in a csv file
#--> add your Python code here

training_file = "weather_training.csv"

with open(training_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for row in reader:
        dbTraining.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here

outlook_mapping = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature_mapping = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity_mapping = {"High": 1, "Normal": 2}
wind_mapping = {"Weak": 1, "Strong": 2}
play_mapping = {"Yes": 1, "No": 2}

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

X = []
Y = []

for row in dbTraining:
    X.append([
        outlook_mapping[row[1]],
        temperature_mapping[row[2]],
        humidity_mapping[row[3]],
        wind_mapping[row[4]]
    ])
    Y.append(play_mapping[row[5]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here

test_file = "weather_test.csv"

dbTest = []

with open(test_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    test_header = next(reader)  # Read test header
    for row in reader:
        dbTest.append(row)

#Printing the header os the solution
#--> add your Python code here

print("Day\tOutlook\tTemperature\tHumidity\tWind\tPlayTennis\tConfidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

for test_instance in dbTest:
    testSample = [
        outlook_mapping[test_instance[1]],
        temperature_mapping[test_instance[2]],
        humidity_mapping[test_instance[3]],
        wind_mapping[test_instance[4]]
    ]

    class_predicted = clf.predict([testSample])[0]
    prob_distribution = clf.predict_proba([testSample])[0]

    confidence = max(prob_distribution)

    if confidence >= 0.75:
        predicted_label = [key for key, value in play_mapping.items() if value == class_predicted][0]
        print(f"{test_instance[0]}\t{test_instance[1]}\t{test_instance[2]}\t{test_instance[3]}\t{test_instance[4]}\t{predicted_label}\t{confidence:.2f}")