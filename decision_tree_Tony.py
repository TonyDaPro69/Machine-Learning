#-------------------------------------------------------------------------
# AUTHOR: Tony Le
# FILENAME: decision_tree_Tony
# SPECIFICATION: This programs uses derivations from the standard ID3 algorithm to create a decision tree.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 8 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

# Encoding dictionaries
age_mapping = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_mapping = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_mapping = {'No': 1, 'Yes': 2}
tear_mapping = {'Reduced': 1, 'Normal': 2}
class_mapping = {'Yes': 1, 'No': 2}

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
#--> add your Python code here
for row in db:
    X.append([
        age_mapping[row[0]],
        spectacle_mapping[row[1]],
        astigmatism_mapping[row[2]],
        tear_mapping[row[3]]
    ])

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
#--> addd your Python code here
for row in db:
    Y.append(class_mapping[row[4]])

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

# Function to calculate entropy
def entropy(labels):
    total = len(labels)
    counts = {label: labels.count(label) for label in set(labels)}
    return -sum((count/total) * math.log2(count/total) for count in counts.values())

# Function to calculate information gain
def information_gain(parent, subsets):
    total = len(parent)
    subset_entropy = sum((len(subset)/total) * entropy(subset) for subset in subsets)
    return entropy(parent) - subset_entropy


#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()