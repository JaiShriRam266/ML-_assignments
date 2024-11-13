# Assignment 3 : Classification technique
# Problem Statement: Every year many students give the GRE exam to get
admission in foreign Universities. The data
# set contains GRE Scores (out of 340), TOEFL Scores (out of 120),
University Rating (out of 5)
# Undergraduate GPA (out of 10), Research Experience (0=no, 1=yes),
Admitted (0=no, 1=yes).
# Admitted is the target variable (i.e. class column) .
# Data Set Available on kaggle (The last column of the dataset needs to
be changed to 0 or 1
# Data Set : https://www.kaggle.com/mohansacharya/graduate-admissions
# The counselor of the firm is supposed check whether the student will
get an admission or not
# based on his/her GRE score and Academic Score. So to help the
counselor to take appropriate
# decisions build a machine learning model classifier using Decision
tree to predict whether a
# student will get admission or not.
# Apply Data pre-processing (Label Encoding, Data Transformation….)
techniques ifnecessary.
# Perform data-preparation (Train-Test Split)
# Evaluate Model.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("Admission_Predict.csv")
# first 5 instances of dataset
data.head()
data.shape
# columns in dataframe
data.columns
# droping Id column as this column is not informative for
classification algorithm
data.drop(
 "Serial No.", axis=1, inplace=True
) # axix = 1 : means said operation is down the rows
# convert the "Chance of Admit" (class column to 0 or 1)
data["Chance of Admit "] = data["Chance of Admit "].apply(
 lambda x: 1 if x > 0.5 else 0
)
# Find missing values
print("Missing values:\n")
data.isnull().sum()
# Calculating total class Count
data_admit = data[data["Chance of Admit "] == 1]
data_non_admit = data[data["Chance of Admit "] == 0]
print("Admitted count : ", data_admit.shape[0])
print("Non - Admitted count : ", data_non_admit.shape[0])
# Vertically Splitting of Dataset into dependent and independent :
input and target (class)
X = data.drop("Chance of Admit ", axis=1)
y = data["Chance of Admit "]
# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# Shape of train Test Split
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
# fit the model : training on data
tree.fit(X_train, y_train)
# predicting the target value from the model for the samples
y_train_tree = tree.predict(X_train)
y_test_tree = tree.predict(X_test)
# computing the accuracy of the model performance
from sklearn.metrics import accuracy_score
acc_test_tree = accuracy_score(y_test, y_test_tree)
print("Decision Tree : Accuracy on test Data:
{:.3f}".format(acc_test_tree))
# computing the classification report of the model
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_tree))
# visualization of tree
import sklearn.tree as tr
fig = plt.figure(figsize=(20, 15))
_ = tr.plot_tree(
 tree,
 feature_names=X.columns,
 class_names=np.array(["Non admit", "Admit"]),
 filled=True,
)
