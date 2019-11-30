import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import random
import math
import keyboard

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

def getXY(filename):
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
	num_samples, num_features = np.shape(X)
	print(f"X num_features: {num_features}")
	fig, axs = plt.subplots(num_features, sharex=True, sharey=True)
	fig.suptitle("Single Feature")
	for i in range(num_features):
		one, two = getFeatureSingle(X, Y, i)
		axs[i].plot(one, np.zeros_like(one), 'bo',  two, np.zeros_like(two), 'r.')
	for i, ax in enumerate(axs.flat):
		ax.set(ylabel=("F"+str(i+1)), xlabel="values")
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
			axs[r][c].text(-3, -4.5, f"{i+1},{j+1}", fontsize=8)
			c += 1
			if c >= cols:
				r += 1
				c = 0
				#plt.show()

	for i, ax in enumerate(axs.flat):
		ax.set(ylabel="", xlabel="", yticks=[], xticks=[])
	
	plt.show()
	pass

def showAllFeatureTriples(X, Y):
	fig = plt.figure()
	axs = fig.add_subplot(222, projection='3d')

	one_f1, one_f2, one_f3, two_f1, two_f2, two_f3 = getFeatureTriple(X,Y, 0, 1, 2)
	axs.scatter(one_f1, one_f2, one_f3, c='b', marker="o")
	axs.scatter(two_f1, two_f2, two_f3, c='r', marker="x")
	plt.show()

def squared_distance(sample, other, features):
	#print(f"_distance :: {features} {len(sample)} {len(other)}")
	squared_distance = 0
	for i in features:
		squared_distance += (sample[i] - other[i])**2
	return squared_distance
	#return np.linalg.norm(sample - other)

def leave_one_out_cross_validation(X, Y, feature_indexes, potential_feature):
	num_correct = 0
	for i, sample in enumerate(X):
		smallest_distance = math.inf
		other_index = -1
		for j, other in enumerate(X):
			if i != j:
				distance = squared_distance(X[i], X[j], feature_indexes+[potential_feature])
				if distance < smallest_distance:
					smallest_distance = distance
					other_index = j
				pass
		if Y[i] == Y[other_index]:
			num_correct += 1

	return num_correct / np.shape(X)[0]
	# return random.random()

def feature_search_demo(X, Y):
	num_samples, num_features = np.shape(X)
	best_features = [] # List of features, ordered from best to worst (using greedy search based on accuracy)
	accuracy_list = [] # List of the feature's corresponding accuracies

	for i in range(num_features):

		print(f"On the {i}th level of the search tree. Best features: {list(map(lambda x: x+1, best_features))} \n")
		best_accuracy = 0
		best_accuracy_feature = 0
		for k in range(num_features):
			if keyboard.is_pressed('space'):
				showFeatureAccuracies(best_features, accuracy_list, True)
			if not k in best_features:
				accuracy = leave_one_out_cross_validation(X, Y, best_features, k)
				print(f"  Considering adding feature [{k+1}] with accuracy: {round(accuracy,2)}")
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					best_accuracy_feature = k

		#best_features.add(most_accurate_feature)
		best_features.append(best_accuracy_feature)
		accuracy_list.append(best_accuracy)
		print(f"  On the {i}th level, added feature [{best_accuracy_feature+1}] with accuracy: {round(best_accuracy, 2)} \n")

	return best_features, accuracy_list

def showFeatureAccuracies(feature_list, accuracy_list, paused=False):
	feature_list = list(map(lambda x: x+1, feature_list))
	accuracy_list = list(map(lambda x: round(x,2), accuracy_list))

	if paused:
		print("\nPaused algorithm. Best features so far:")
	matrix = np.array([feature_list, accuracy_list]).transpose()
	print(matrix)
	print()

	plt.plot(range(len(feature_list)), accuracy_list, 'o-')
	plt.xticks(range(len(feature_list)), feature_list)
	plt.yticks([(i)*0.1 for i in range(11)])
	plt.xlabel("features")
	plt.ylabel("accuracy")

	for x,y in zip(range(10), accuracy_list):
		label = "{:.2f}".format(y)
		plt.annotate(label,
			(x,y),
			textcoords="offset points",
			xytext=(0,10),
			ha='center')

	plt.show()

def main():
	#filename = 'small_38.txt'
	filename = 'large_40.txt'
	X, Y = getXY(filename)
	X = np.array(X)
	Y = np.array(Y).transpose()
	Y = np.expand_dims(Y, axis=1) # convert into a (300x1) matrix instead of a (300,)

	f1, f2 = 44, 59
	one_f1, one_f2, two_f1, two_f2 = getFeaturePair(X, Y, f1-1, f2-1)
	plt.plot(one_f1, one_f2, 'b.', markersize=5)
	plt.plot(two_f1, two_f2, 'r.', markersize=5)
	plt.xlabel(f"Feature {f1}")
	plt.ylabel(f"Feature {f2}")
	plt.show()

	#showAllFeatureSingles(X, Y)
	#showAllFeaturePairs(X,Y, 10)

	best_features, accuracy_list = feature_search_demo(X, Y)
	print("Done!")
	showFeatureAccuracies(best_features, accuracy_list)

if __name__ == '__main__':
	main()
	