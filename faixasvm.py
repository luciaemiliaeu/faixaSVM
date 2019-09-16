import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 

def acuracia( min, max, attr):
	acertos = 0
	for i in attr:
		if i>=min and i<=max: acertos += 1
	score = acertos / attr.shape[0]
	return score

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
	
	clf = OneVsRestClassifier(SVC( gamma=.5, C=0.1))
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
			print(nomes[i], min[i], " - ", max[i], " ", acuracia(min[i], max[i], data.loc[:,data.columns[i]]))


