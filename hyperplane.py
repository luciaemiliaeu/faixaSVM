import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from sklearn import datasets

def plane(xx, w):
	aux = []
	for x in xx:
		aux.append(sum(w.T*x))
	return aux

def plot_hyperplane(clf, map_ , XX, linestyle, label):
	print(label)
	# get the separating hyperplane
	w = clf.coef_[0]
	a = -w[:-1]/w[-1]
	b = clf.intercept_/w[-1]
	#z = map_[:,:-1] * a - b
	z = plane(map_[:,:-1], a) - b
	
	# plot the line, the points, and the nearest vectors to the plane
	ax.plot(map_[:,0], map_[:,1], z, label = label)



iris = datasets.load_iris()
X = iris.data[:,:3]
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
Y = kmeans.labels_

attr1_min = X[:,0].min()
attr1_max = X[:,0].max()
attr2_min = X[:,1].min()
attr2_max = X[:,1].max()
attr3_min = X[:,2].min()
attr3_max = X[:,2].max()

xx = np.linspace(attr1_min, attr1_max)
yy = np.linspace(attr2_min, attr2_max)
zz = np.linspace(attr3_min, attr3_max)

XX, YY, ZZ = np.meshgrid(xx,yy,zz)

map_ = np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()]

clf = OneVsRestClassifier(SVC(kernel='linear'))

clf.fit(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[2])
ax.set_zlabel(iris.feature_names[3])

colors = ['tab:blue', 'tab:orange', 'tab:green']
for y, c in zip(np.unique(Y), colors):
	x = X[Y==y]
	ax.scatter(x[:,0], x[:,1], x[:,2], s=40, c = c, facecolors= 'none')

ax.scatter(clf.estimators_[0].support_vectors_[:,0], clf.estimators_[0].support_vectors_[:,1],clf.estimators_[0].support_vectors_[:,2],
	s=100, edgecolors='k',linewidths=2, facecolors='none')

titulos = ["Boundary for cluster 1", "Boundary for cluster 2", "Boundary for cluster 3"]
for c, label  in zip(clf.estimators_, titulos):
	plot_hyperplane(c,map_, XX, 'k--', label)

ax.legend()
plt.show()