# PLEASE WRITE THE GITHUB URL BELOW!
# 

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm
from sklearn.metrics import confusion_matrix


def load_dataset(dataset_path):
	#To-Do: Implement this function
	data = pd.read_csv(dataset_path)
	return data

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
    nFeature = len(dataset_df.columns) - 1
    nClass0 = len(dataset_df.loc[dataset_df['target'] == 0])
    nClass1 = len(dataset_df.loc[dataset_df['target'] == 1])
    return nFeature, nClass0, nClass1
 
def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
    X = dataset_df.drop(columns="target", axis=1)
    y = dataset_df["target"]
    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=testset_size)
    return train_data, test_data, train_label, test_label
 
def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    
    dtc = DecisionTreeClassifier()
    
    dtc.fit(x_train, y_train)
    
    return accuracy_score(y_test, dtc.predict(x_test)), precision_score(y_test, dtc.predict(x_test)), recall_score(y_test, dtc.predict(x_test))
	
 
def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    rf = RandomForestClassifier()
    
    rf.fit(x_train, y_train)
    
    return accuracy_score(y_test, rf.predict(x_test)), precision_score(y_test, rf.predict(x_test)), recall_score(y_test, rf.predict(x_test))
	
 
def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    pipe = make_pipeline(
        StandardScaler(),
        svm.SVC()
    )
    
    pipe.fit(x_train, y_train)
    
    return accuracy_score(y_test, pipe.predict(x_test)), precision_score(y_test, pipe.predict(x_test)), recall_score(y_test, pipe.predict(x_test))
 
def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)
	
if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)