from data_utils import load_dataset
import math
import numpy
import matplotlib.pyplot as plt
import time
import heapq
import sklearn.neighbors


# regression_variations.py: Evaluates the computational complexity of kNN algorithm variations

# Calculates Root Mean Square for two input vectors x1 and x2
def rmse(x1, x2):
    result = 0
    for i in range(len(x1)):
        result += (x1[i] - x2[i]) * (x1[i] - x2[i])
    result = result/len(x1)
    return math.sqrt(result)


# Calculates the distance between two input vectors x1 and x2 according to the distance metric specified as type
def distance(type, x1, x2):
    if type == "L1":
        result = 0
        for i in range(len(x1)) :
                result += abs(x1[i]-x2[i])
        return result
    if type == "L2":
        result = 0
        for i in range(len(x1)):
            result += (x1[i] - x2[i])*(x1[i] - x2[i])
        return math.sqrt(result)
    if type == "L00":
        result = 0
        for i in range(len(x1)):
            val = abs(x1[i] - x2[i])
            if val > result:
                result = val
        return result
    else:
        print("ERROR: Unrecognized distance metric")
        return 0


# Brute Force implementation of the get neighbors algorithm, copied from part one verbatim
def get_neighbors_brute(distancetype, xarray, centerpoint, k):
    distances = []
    if isinstance(centerpoint,list):
        centerpoint = centerpoint[0]
    elif isinstance(centerpoint,numpy.float64):
        centerpoint = [centerpoint]

    #For Loop over Training Set
    for i in range(len(xarray)):
        if isinstance(xarray[i], numpy.float64):
            arr = [xarray[i]]
        else:
            arr = xarray[i]
        temp = distance(distancetype, centerpoint, arr)
        distances.append((temp, i))
        if not isinstance(temp, numpy.float64) and not isinstance(temp, float):
            print("ERROR")
            print(type(temp))
            print(temp)
            print(centerpoint)
            print(xarray[i])

    distances.sort(key=lambda tup: tup[0])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][1])

    return neighbors


# Brute Force implementation of the kNN, copied from part one verbatim
def knn_brute(xarray, yarray, distanceType, centerpoint, k):

    neighbors = get_neighbors_brute(distanceType, xarray, centerpoint, k)
    tot = 0
    for ind in neighbors:
        tot += yarray[ind]
    result = tot/k

    return result


# Partial Vectorization implementation of the get neighbors function, which passes the entire testing array at once
def partialvec_get_neighbors(xarray, centerpoint, k):

    # Calculates distance by matrix operation, and uses heapq to obtain the indices of the n smallest values
    distances = numpy.sqrt(numpy.sum(numpy.square(xarray-centerpoint), axis=1))
    neighbors = heapq.nsmallest(k, range(len(distances)), distances.take)

    return neighbors


# Partial Vectorization implementation of kNN, same as brute force but calls a different get neighbors function
def partialvec_knn(xtrain, ytrain, xpoint, k):

    neighbors = partialvec_get_neighbors(xtrain, xpoint, k)

    tot = 0
    for ind in neighbors:
        tot += ytrain[ind]
    result = tot/k

    return result


# KD Tree Implementation of the get neighbors function, passing the entire testing and training array at once
def kdtree_get_neighbors(xtrain, xtest, k):

    # Generates a KD tree data structure and queries it to find the indices of the k nearest neighbors for each
    # entry to the testing set
    tree = sklearn.neighbors.KDTree(xtrain)
    dist,ind = tree.query(xtest, k=k)

    return ind


# KD Tree Implementation of the kNN algorithm, passing the entire training set and testing set at once
def kdtree_knn(xtrain, ytrain, xtest, k):

    neighbors = kdtree_get_neighbors(xtrain, xtest, k)
    # Uses matrix operations to take the values from the closest neighbors array, and average the values at every row
    # to create the estimate array
    estimate = numpy.sum(numpy.matrix.take(ytrain, neighbors), axis=1) / k

    return estimate


# Fully Vectorized get neighbors implementation, passing the entire training set and testing set at once
def vec_get_neighbors(train_set, test_set, k):

    # Uses matrix operations to calculate a matrix of the indices of the closest neighbors for each point in the testing set
    distance_mat = numpy.sqrt(-2*numpy.dot(test_set, train_set.T) + numpy.sum(train_set**2, axis=1) + numpy.sum(test_set**2, axis=1)[:, numpy.newaxis])
    neighbors = numpy.argpartition(distance_mat, kth = k, axis=1)[:,:k]

    return neighbors


# Fully vectorized implementation of kNN using fully vetorized get neighbors function and passing the entire training and testing sets
def vec_knn(xtrain, ytrain, xtest, k):

    neighbors = vec_get_neighbors(xtrain, xtest, k)
    # Uses matrix operations to take the values from the closest neighbors array, and average the values at every row
    # to create the estimate array
    estimate = numpy.sum(numpy.matrix.take(ytrain, neighbors), axis=1)/k

    return estimate


if __name__ == '__main__':

    print("_________________________________________________________________________________________________________")
    print("Brute Force")
    times = []
    error = []

    # Calculates the runtime of the algorithm and RMSE error as the algorithm runs on the Rosenbrock set for increasing values of d
    for d in range(9):
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("rosenbrock", n_train=5000, d=2+d)

        k, distance_type = [5, "L2"]

        start = time.time()
        estimate = []
        for j in range(len(x_test)):
            estimate.append(knn_brute(x_train, y_train, distance_type, [x_test[j]], k))
        end = time.time()
        times.append(end - start)
        error.append(rmse(estimate, y_test))
        print("Done Iteration")

    # Generates plots of runtime and RMSE error with respect to d value
    plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], times)
    plt.show()
    plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], error)
    plt.show()

    print("_________________________________________________________________________________________________________")
    print("Partially Vectorized")
    times = []
    error = []
    for d in range(9):
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("rosenbrock", n_train=5000, d=2 + d)

        start = time.time()
        estimate = []
        # For Loop over Testing Set, see getneighbors() function for For Loop over Training Set
        for j in range(len(x_test)):
            estimate.append(partialvec_knn(x_train, y_train, x_test[j], 5))
        end = time.time()
        times.append(end - start)
        error.append(rmse(estimate, y_test))

    plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], times)
    plt.show()
    plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], error)
    plt.show()


    print("_________________________________________________________________________________________________________")
    print("Fully Vectorized")
    times = []
    error = []
    for d in range(9):
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("rosenbrock", n_train=5000, d=2 + d)

        start = time.time()
        estimate = []
        # For Loop over Testing Set, see getneighbors() function for For Loop over Training Set
        start = time.time()
        estimate = []
        estimate = vec_knn(x_train, y_train, x_test, 5)
        end = time.time()
        times.append(end-start)
        error.append(rmse(estimate, y_test))

    plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], times)
    plt.show()
    plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], error)
    plt.show()


    print("_________________________________________________________________________________________________________")
    print("K-D Tree")
    times = []
    error=[]
    for d in range(9):
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("rosenbrock", n_train = 5000, d=2+d)

        start = time.time()
        estimate = []
        estimate = kdtree_knn(x_train, y_train, x_test, 5)
        end = time.time()
        times.append(end-start)
        error.append(rmse(estimate,y_test))


    plt.plot([2,3,4,5,6,7,8,9,10],times)
    plt.show()
    plt.plot([2,3,4,5,6,7,8,9,10],error)
    plt.show()
