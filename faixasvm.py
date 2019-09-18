import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 

def plotting(title, axis,Y, X, estimators):
	# Configurações do gráfico
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel(axis[0])
	ax.set_ylabel(axis[1])
	ax.set_zlabel(axis[2])
	plt.title(title)
	
	# Plota os pontos
	colors = plt.get_cmap('Accent').colors	
	for y, c in zip(np.unique(Y), colors):
		x = X[Y==y]
		ax.scatter(x[:,0], x[:,1], x[:,2], s=40, c = [c], facecolors= 'none', label=("Cluster "+str(y)))

	# Plota os vetores de suporte para cada classe
	for i in estimators:
		ax.scatter(i.support_vectors_[:,0], i.support_vectors_[:,1], i.support_vectors_[:,2],
			s=100, edgecolors='k',linewidths=2, facecolors='none')
	ax.legend()

def acuracia( min, max, attr):
	acertos = 0
	for i in attr:
		if i>=min and i<=max: acertos += 1
	score = acertos / attr.shape[0]
	return score

chamadas = [("./databases/iris.csv",3),("./databases/vidros.csv",7), ("./databases/sementes.csv",3)]

for dataset, n_clusters in chamadas:
	# Extrai o nome da base de dados
	title = dataset.split('/')[2].split('.')[0]+' dataset'
	print("")
	print(title)
	print("")

	# Cria DataFrame com os valores de X e o cluster Y
	cluster = pd.read_csv(dataset, sep=',',parse_dates=True)
	cluster = cluster.drop('classe', axis=1)
	X = cluster.get_values()
	Y = KMeans(n_clusters=n_clusters, random_state=0).fit(X).labels_
	cluster.loc[:,'Cluster'] = pd.Series(Y)
	
	# Treina o classificador com todas as amostras
	clf = OneVsRestClassifier(SVC( gamma=.5, C=0.9))
	clf.fit(X,Y)

	# Separa os grupos em Frames
	clusters = cluster.groupby(['Cluster']) 
	nomes = cluster.columns

	# Para cada grupo: 
	for n_grupo, data in clusters:
		# Seleciona os vetores de suprote que separam aquele grupo
		vetores = pd.DataFrame(columns=range(X.shape[1]+1))
		c = clf.estimators_[n_grupo]
		for  indice in  c.support_:
			if Y[indice] == n_grupo:
				vetores.loc[vetores.shape[0]]= pd.Series(np.append(X[indice,:], [Y[indice]]))
		
		# Calcula o min e max de cada atributo
		min = vetores.min(axis=0)
		max = vetores.max(axis=0)
		
		#Monta o rótulo de cada grupo
		print("\n", "Grupo ", n_grupo, "#Elementos", data.shape[0])
		for i in range(X.shape[1]):
			print(nomes[i], min[i], " - ", max[i], " ", acuracia(min[i], max[i], data.loc[:,data.columns[i]]))
	
	# Plota a base de dados em 3D, realçando os vetores de suporte
	plotting(title, nomes, Y, X, clf.estimators_)
plt.show()