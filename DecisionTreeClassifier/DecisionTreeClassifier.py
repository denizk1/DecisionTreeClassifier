import numpy as np
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections
from sklearn.datasets import load_iris

data=load_iris() #We are getting data from sklearn
X ,y ,data_names= data["data"], data["target"], data["feature_names"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
dot_data = tree.export_graphviz(clf,
                                feature_names=data_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')