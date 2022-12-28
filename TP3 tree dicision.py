import pandas as pd
import  numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import graphviz 
from sklearn.datasets import load_iris
iris = load_iris()



iris_data = pd.read_csv("iris.csv")
iris_data.head()


clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None,feature_names=iris.feature_names,class_names=iris.target_names,filled=True, rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data)


graph

#save in to a pdf file
graph.render("iris")