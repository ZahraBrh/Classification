import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import  scale
import  sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report
from mpl_toolkits.mplot3d import  Axes3D

import scipy 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import  fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import distance 
from pylab import rcParams
import seaborn as sb


data = pd.read_csv('bdd.csv',header=None)
data.drop(31,1,inplace=True)
print(data.shape)

print(data.describe())
#CAH 
Z = linkage(data,method='ward',metric='euclidean')
dendrogram(Z,truncate_mode='lastp',p=26,leaf_rotation=45,leaf_font_size=10,show_contracted=True)
#dendrogram(Z,labels=data.index,color_threshold=800,orientation="right")
plt.title('CAH')
plt.xlabel('Cluster Size')
plt.ylabel('distance')
plt.axhline(y=500)
plt.axhline(y=150) #le trait dans le dendrograme
plt.show()

clusters = fcluster(Z,t=800, criterion='distance')
print(clusters)

idg = np.argsort(clusters)
print(pd.DataFrame(data.index[idg],clusters[idg]))



