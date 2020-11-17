from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


wine = load_wine()
#print(data.DESCR)

cols=["Alcohol","Malic acid","Ash","Alcalionity","Magnesium","Total Phenol","Flavanoids",
                                             "Nonflavor","Proanthocyanins","Color intensity","Hue","OD280_OD315","Proline"]
df=pd.DataFrame(wine["data"][:,0:13],columns=cols)
'''

		- Alcalinity of ash  
 		- Magnesium
		- Total phenols
 		- Flavanoids
 		- Nonflavanoid phenols
 		- Proanthocyanins
		- Color intensity
 		- Hue
 		- OD280/OD315 of diluted wines
 		- Proline])
'''
print(df)
maxi=df.max()
mini=df.min()
print(maxi,mini)
df2=(df-mini)
delta=maxi-mini
df_feature=df2/delta
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
import sklearn.cluster as cluster


from sklearn.decomposition import PCA

n_cluster_num=3
clusterer = KMeans(n_clusters=n_cluster_num, random_state=10)
cluster_labels_Kmeans = clusterer.fit_predict(df_feature)

clusterer=DBSCAN(eps=0.45)
cluster_label_DBScan=clusterer.fit_predict(df_feature)

clusterer=Birch(n_clusters=n_cluster_num)
cluster_label_Birch=clusterer.fit_predict(df_feature)

bandwidth = cluster.estimate_bandwidth(df_feature, quantile=0.15)
clusterer=MeanShift(bin_seeding=True,bandwidth=bandwidth)
cluster_label_MeanShift=clusterer.fit_predict(df_feature)



a_pca=PCA(n_components=3)
data_pca=a_pca.fit_transform(df_feature)

Y=wine.target

# Kezdjünk új ábrát (plt.figure)!
plt.figure(figsize=(20,5))
# Rajzoljunk a plt.scatter segítségével!
# Segítség: X_pca[:, 0], X_pca[:, 1], c=Y
plt.subplot(151)
plt.xlabel("Kmeans")
plt.scatter(data_pca[:,0],data_pca[:,1],c=cluster_labels_Kmeans)
# Állítsuk be a tengelyek címkéit és a címet!
plt.subplot(152)
plt.xlabel("DBScan")
plt.scatter(data_pca[:,0],data_pca[:,1],c=cluster_label_DBScan)

plt.subplot(153)
plt.xlabel("Birch")
plt.scatter(data_pca[:,0],data_pca[:,1],c=cluster_label_Birch)

plt.subplot(154)

plt.xlabel("MeanShift")
plt.scatter(data_pca[:,0],data_pca[:,1],c=cluster_label_MeanShift)

plt.subplot(155)
plt.xlabel("PCA Y")
plt.scatter(data_pca[:,0],data_pca[:,1],c=Y)
...
# Jelenítsük meg a plt.show metódus segítségével!
plt.show()

