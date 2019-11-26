import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import random
import math

def classify_knn(datapoint, X, Y, k=1):
	_class = Y[0]
	closest_distance = math.inf
	for i, sample in enumerate(X):
		distance = np.linalg.norm(datapoint - sample)
		if distance < closest_distance:
			closest_distance = distance
			_class = Y[i]
	return _class

def forward_selection():

	pass

def IEEE754_num(ieee754):
	number, exponent = ieee754.split('e')
	return float(number) * (10 ** int(exponent))

def get_csv_as_matrix(filename):
	X, Y = [], []
	with open(filename, 'r') as file:
		csv_reader = csv.reader(file, delimiter='\n')
		for line in csv_reader:
			row = []
			line = line[0].split()
			for i, number in enumerate(line):
				num = IEEE754_num(number)
				if i == 0:
					Y.append(int(num))
				else:
					row.append(num)
			X.append(row)
	return X, Y

def getFeatureSingle(X, Y, f1):
	class1_f1, class2_f1 = [], []
	for i, sample in enumerate(X):
		if Y[i] == 1:
			class1_f1.append(sample[f1])
		elif Y[i] == 2:
			class2_f1.append(sample[f1])
	return class1_f1, class2_f1

def getFeaturePair(X, Y, f1, f2):
	class1_f1, class1_f2, class2_f1, class2_f2 = [], [], [], []
	for i, sample in enumerate(X):
		if Y[i] == 1:
			class1_f1.append(sample[f1])
			class1_f2.append(sample[f2])
		elif Y[i] == 2:
			class2_f1.append(sample[f1])
			class2_f2.append(sample[f2])
	return class1_f1, class1_f2, class2_f1, class2_f2

def getFeatureTriple(X, Y, f1, f2, f3):
	one_f1, one_f2, one_f3, two_f1, two_f2, two_f3 = [], [], [], [], [], []
	for i, sample in enumerate(X):
		if Y[i] == 1:
			one_f1.append(sample[f1])
			one_f2.append(sample[f2])
			one_f3.append(sample[f3])
		elif Y[i] == 2:
			two_f1.append(sample[f1])
			two_f2.append(sample[f2])
			two_f3.append(sample[f3])
	return one_f1, one_f2, one_f3, two_f1, two_f2, two_f3

def showAllFeatureSingles(X,Y):
	fig, axs = plt.subplots(10, sharex=True, sharey=True)
	fig.suptitle("Single Feature")
	for i in range(10):
		one, two = getFeatureSingle(X, Y, i)
		axs[i].plot(one, np.zeros_like(one), 'bo',  two, np.zeros_like(two), 'r.')
	for i, ax in enumerate(axs.flat):
		ax.set(ylabel=("f"+str(i)), xlabel="feature values")
	plt.yticks([])
	plt.show()

def showAllFeaturePairs(X,Y, num_features):
	rows, cols = 9, 5
	r, c = 0, 0
	fig, axs = plt.subplots(rows, cols)
	fig.suptitle("Feature Pairs")
	for i in range(0, num_features-1):
		for j in range(i+1, num_features):
			one_f1, one_f2, two_f1, two_f2 = getFeaturePair(X, Y, i, j)
			axs[r][c].plot(one_f1, one_f2, 'b.', markersize=1)
			axs[r][c].plot(two_f1, two_f2, 'r.', markersize=0.5)
			axs[r][c].set_xlim(-3,3)
			axs[r][c].set_ylim(-5,5)
			axs[r][c].text(-3, -4.5, f"{i},{j}", fontsize=8)
			c += 1
			if c >= cols:
				r += 1
				c = 0
				#plt.show()

	for i, ax in enumerate(axs.flat):
		ax.set(ylabel="", xlabel="", yticks=[], xticks=[])
	
	plt.show()
	pass

def showAllFeatureTriples(X,Y):
	fig = plt.figure()
	axs = fig.add_subplot(222, projection='3d')

	one_f1, one_f2, one_f3, two_f1, two_f2, two_f3 = getFeatureTriple(X,Y, 0, 1, 2)
	axs.scatter(one_f1, one_f2, one_f3, c='b', marker="o")
	axs.scatter(two_f1, two_f2, two_f3, c='r', marker="x")
	plt.show()

def main():
	X, Y = get_csv_as_matrix('small_38.txt')
	X = np.array(X)
	Y = np.array(Y).transpose()
	Y = np.expand_dims(Y, axis=1) # convert into a (300x1) matrix instead of a (300,)

	datapoint = X[random.randint(0,300)]
	_class = classify_knn(datapoint, X, Y)
	print("Result: ")
	print(X[0], _class)

	#showAllFeatureSingles(X, Y)
	showAllFeaturePairs(X,Y, 10)

	

if __name__ == '__main__':
	main()
	