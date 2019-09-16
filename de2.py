import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from sklearn import datasets

def plane(xx, w):
	print("D: ", xx.shape)
	print("Ws: ", w.shape)
	aux = []
	for x in xx:
		aux.append(sum(w.T*x))
	return aux

def plot_hyperplane(clf, map_ , linestyle):
	
	# get the separating hyperplane
	w = clf.coef_[0]
	a = -w[:-1]/w[-1]
	b = clf.intercept_/w[-1]
	z = plane(map_[:,:-1], a) - b

	# plot the line, the points, and the nearest vectors to the plane
	ax.plot(map_[:,:-1], z, 'k-')


iris = datasets.load_iris()
X = iris.data
Y = iris.target

attr1_min = X[:,0].min()
attr1_max = X[:,0].max()
attr2_min = X[:,1].min()
attr2_max = X[:,1].max()

xx = np.linspace(attr1_min, attr1_max)
yy = np.linspace(attr2_min, attr2_max)
XX, YY = np.meshgrid(xx,yy)
map_ = np.c_[XX.ravel(), YY.ravel()]


clf = OneVsRestClassifier(SVC(gamma=.5, C=0.1))

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

ax.legend()
plt.show()