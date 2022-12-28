#librairies
import pandas
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage , fcluster
import numpy as np
from sklearn import cluster
from sklearn import metrics

#chargement des données
data = pandas.read_csv("bdd.csv",header=None)
data.drop(31,1,inplace=True)
print(data.shape)

#statistiques descriptives
print(data.describe())

#graphique - croisement deux à deux des variables
#from pandas.plotting import scatter_matrix
#scatter_matrix(fromage,figsize=(9,9))



#génération de la matrice des liens
Z = linkage(data,method='ward',metric='euclidean')

#affichage du dendrogramme
plt.title('CAH avec matérialisation des 26 classes')
dendrogram(Z,labels=data.index,color_threshold=800,orientation="right")
plt.show()

#découpage en classes
groupes_cah = fcluster(Z,t=800,criterion='distance')
print(groupes_cah)



#affichage des observations et leurs groupes
idg = np.argsort(groupes_cah)
print(pandas.DataFrame(data.index[idg],groupes_cah[idg]))



#k-means sur les données

kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(data)
idk = np.argsort(kmeans.labels_)

#affichage des observations et leurs groupes
print(pandas.DataFrame(data.index[idk],kmeans.labels_[idk]))

#distances aux centres de classes des observations
print(kmeans.transform(data))

#correspondance avec les groupes de la CAH
pandas.crosstab(groupes_cah,kmeans.labels_)

#utilisation de la métrique "silhouette"
res = np.arange(9,dtype="double")
for k in np.arange(9):
   km = cluster.KMeans(n_clusters=k+2)
   km.fit(data)
   res[k] = metrics.silhouette_score(data,km.labels_)
print(res)
plt.title("Silhouette")
plt.xlabel("# of clusters")
plt.plot(np.arange(2,11,1),res)
plt.show()