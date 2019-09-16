import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from sklearn import datasets

chamadas = [("./databases/iris.csv",3),("./databases/vidros.csv",7), ("./databases/sementes.csv",3)]

for dataset, n_clusters in chamadas:
	print("")
	print(dataset)
	print("")
	cluster = pd.read_csv(dataset, sep=',',parse_dates=True)
	cluster = cluster.drop('classe', axis=1)
	X = cluster.get_values()
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
	Y = kmeans.labels_
	
	cluster.loc[:,'Cluster'] = pd.Series(Y)
	clusters = cluster.groupby(['Cluster']) 
	nomes = cluster.columns
	
	clf = OneVsRestClassifier(SVC(kernel='linear'))
	clf.fit(X,Y)


	for grupo, data in clusters:
		vetores = pd.DataFrame(columns=range(X.shape[1]+1))
		c = clf.estimators_[grupo]
		for  indice in  c.support_:
			if Y[indice] == grupo:
				vetores.loc[vetores.shape[0]]= pd.Series(np.append(X[indice,:], [Y[indice]]))
		print("\n", "Grupo ", grupo, "#Elementos", data.shape[0])
		min = vetores.min(axis=0)
		max = vetores.max(axis=0)
		for i in range(X.shape[1]):
			print(nomes[i], min[i], " - ", max[i])

