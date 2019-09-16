import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from sklearn import datasets

def plane(xx, w):
	for x in xx:
		x = sum(w.T*x)
	return xx

def plot_hyperplane(clf, map_ , linestyle):
	
	# get the separating hyperplane
	w = clf.coef_[0]
	a = -w[:-1]/w[-1]
	b = clf.intercept_/w[-1]
	z = plane(map_[:,:-1], a) - b
	
	# plot the parallels to the separating hyperplane that pass through the
	# support vectors (margin away from hyperplane in direction
	# perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
	# 2-d.
	margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
	yy_down = z -  margin
	yy_up = z +  margin

	# plot the line, the points, and the nearest vectors to the plane
	
	ax.plot(map_[:,:-1], z, 'k-')
	#plt.plot(map_[:,0], yy_down, 'k--')
	#plt.plot(map_[:,0], yy_up, 'k--')

iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target

attr1_min = X[:,0].min()
attr1_max = X[:,0].max()
attr2_min = X[:,1].min()
attr2_max = X[:,1].max()

xx = np.linspace(attr1_min, attr1_max)
yy = np.linspace(attr2_min, attr2_max)
XX, YY = np.meshgrid(xx,yy)
map_ = np.c_[XX.ravel(), YY.ravel()]

'''
iris = datasets.load_iris()
X = iris.data[:,:3]
Y = iris.target

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

iris = datasets.load_iris()
X = iris.data[:,:4]
Y = iris.target

attr1_min = X[:,0].min()
attr1_max = X[:,0].max()
attr2_min = X[:,1].min()
attr2_max = X[:,1].max()
attr3_min = X[:,2].min()
attr3_max = X[:,2].max()
attr4_min = X[:,3].min()
attr4_max = X[:,3].max()

xx = np.linspace(attr1_min, attr1_max)
yy = np.linspace(attr2_min, attr2_max)
zz = np.linspace(attr3_min, attr3_max)
kk = np.linspace(attr4_min, attr4_max)

XX, YY, ZZ, KK = np.meshgrid(xx,yy, zz, kk)
map_ = np.c_[XX.ravel(), YY.ravel(), ZZ.ravel(), KK.ravel()]
'''

clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], s=40, c=Y, facecolors= 'face' )
ax.scatter(clf.estimators_[0].support_vectors_[:,0], clf.estimators_[0].support_vectors_[:,1],
	s=100, edgecolors='b',linewidths=2, facecolors='none' )

plot_hyperplane(clf.estimators_[0],map_, 'k--')

plt.show()