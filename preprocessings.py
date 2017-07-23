import numpy as np
import matplotlib.pyplot as plt

def generate_data(nb_points, m, C):
	return np.random.multivariate_normal(m, C, nb_points)

def mean_cancellation(X):
	return X - np.mean(X, axis=0)

def pca(X):
	U, S, V = np.linalg.svd(X)
	return np.dot(X, V.T)

def covariance_equalization(X):
	return X / np.std(X, axis=0)

if __name__ == '__main__':
	X = generate_data(100, np.array([2, 2.5]), np.array([[1.01, 0.99], [0.99, 1.01]]))
	plt.subplot(2, 2, 1)
	plt.plot(X[:,0], X[:,1], '.')
	#plt.axis('equal')
	plt.xlim(-6, 6)
	plt.ylim(-6, 6)
	plt.title('Original data (1)')

	X = mean_cancellation(X)
	plt.subplot(2, 2, 2)
	plt.plot(X[:,0], X[:,1], '.')
	#plt.axis('equal')
	plt.xlim(-6, 6)
	plt.ylim(-6, 6)
	plt.title('Mean cancellation (2)')

	X = pca(X)
	plt.subplot(2, 2, 4)
	plt.plot(X[:,0], X[:,1], '.')
	#plt.axis('equal')
	plt.xlim(-6, 6)
	plt.ylim(-6, 6)
	plt.title('PCA (3)')

	X = covariance_equalization(X)
	plt.subplot(2, 2, 3)
	plt.plot(X[:,0], X[:,1], '.')
	#plt.axis('equal')
	plt.xlim(-6, 6)
	plt.ylim(-6, 6)
	plt.title('Covariance equalization (4)')

	plt.show()