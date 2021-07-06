from data_utils import load_dataset
import numpy, heapq, math
import matplotlib.pyplot as plt


# classification.py: k Nearest Neighbor algorithm for image classification

# Calculates the distance between two input vectors x1 and x2 according to the distance metric specified as type
def distance(dtype, x1, x2):
    if dtype == "L1":
        return numpy.sum(numpy.absolute(x1 - x2), axis=1)
    if dtype == "L2":
        return numpy.sqrt(numpy.sum(numpy.square(x1 - x2), axis=1))
    if dtype == "L00":
        result = 0
        for i in range(len(x1)):
            val = abs(x1[i] - x2[i])
            if val > result:
                result = val
        return result
    else:
        print("ERROR: Unrecognized distance metric")
        return 0


# Returns the indices of the k closest neighbors in "xarray" for the x value "centerpoint" using the partial
# vectorization variation in part 3
def get_neighbors(distancetype, main, reference, k):
    distances = distance(distancetype, main, reference)
    neighbors = heapq.nsmallest(k, range(len(distances)), distances.take)

    return neighbors


# Polls the k closest neighbors and returns the index of the most often recurring class
def knn_estimate (xarray, yarray, distanceType, centerpoint, k):
    neighbors = get_neighbors(distanceType, xarray, centerpoint, k)
    tot = [0]*len(yarray[0])
    for ind in neighbors:
        for i in range(len(yarray[0])):
            if yarray[ind][i]:
                tot[i] += 1
    maxval = 0
    maxind = None
    for i in range(len(yarray[0])):
        if tot[i] > maxval:
            maxval = tot[i]
            maxind = i
    return maxind


# Performs 5 fold Cross Validation for the Classification kNN, returning the optimal k value, distance metric,
# and success rate
def five_fold_cross(x, y):

    # Shuffles the x and y array (ensuring corresponding values have the same index
    merged = []
    for i in range(len(x)):
        merged.append([y[i], x[i]])

    numpy.random.seed(0)
    numpy.random.shuffle(merged)
    y = []
    x = []
    for i in range(len(merged)):
        y.append(merged[i][0])
        x.append(merged[i][1])

    # Splits the data into 5 partitions
    n = math.floor(len(x) / 5)
    x_partitioned = [x[:n], x[(n + 1):(2 * n)], x[(2 * n + 1):(3 * n)], x[(3 * n + 1):(4 * n)], x[(4 * n + 1):]]
    n = math.floor(len(y) / 5)
    y_partitioned = [y[:n], y[(n + 1):(2 * n)], y[(2 * n + 1):(3 * n)], y[(3 * n + 1):(4 * n)], y[(4 * n + 1):]]

    result = [None]*len(y)
    max_rate = 0

    # Iterates through every distance metric, k values from 1 to 9, and performs a 5 fold cross validation for each configuration
    for dist_type in ["L1","L2","L00"]:
        print(dist_type)
        for k_val in range(1, 10):
            rate = 0
            for i in range(5):
                # Every partition is a testing set for an iteration, and the other make up the training set
                x_testing = x_partitioned[i]
                y_testing = y_partitioned[i]
                x_training = []
                y_training = []

                for j in range(5):
                    if i !=j:
                        for element in x_partitioned[j]:
                            x_training.append(element)
                        for element in y_partitioned[j]:
                            y_training.append(element)

                # Builds an array of estimates using the kNN algorithm
                estim = []
                # The ith entry for estim is an array where every value is false except for the estimated class
                # of the ith element of the x array
                for j in range(len(x_testing)):
                    temp = [False]*len(y_testing[0])
                    temp[knn_estimate(x_training, y_training, dist_type, x_testing[j], k_val)] = True
                    estim.append(temp)

                # Calculates Success rate by counting the number of times the True boolean is at the same index for the
                # estimate and the actual y values of the testing set
                correct = 0
                total = len(y_testing)
                for j in range(total):
                    cond = False
                    for l in range(len(y_testing[j])):
                        if y_testing[j][l] and estim[j][l]:
                            cond = True
                    if cond:
                        correct += 1
                rate += correct/total
                print(rate)
            rate = rate/5
            if rate > max_rate:
                max_rate = rate
                result = [max_rate, k_val, dist_type]
            print("Finished Iteration # %d for k value %d and distance metric %s, with rate of %f" % (i, k_val, dist_type, rate))

    return result[0], result[1], result[2]



if __name__ == '__main__':
    print("_________________________________________________________________________________________________________")
    print("Iris Dataset")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("iris")

    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=0)

    cross_best_rate, k, distance_type = five_fold_cross(x_merge, y_merge)

    estimate = []
    for j in range(len(x_test)):
        temp = [False, False, False]
        temp[knn_estimate(x_test, y_test, distance_type, x_test[j], k)] = True
        estimate.append(temp)
    correct = 0
    total = len(y_test)
    for j in range(total):
        if (y_test[j][0] and estimate[j][0]) or (y_test[j][1] and estimate[j][1]) or (y_test[j][2] and estimate[j][2]):
            correct += 1
    test_rate = correct / total
    print("The Optimal Configuration is a K-value of " + str(k)+ " and using the " + distance_type + " distance metric")
    print("The success rate at the optimal configuration in cross validation was " + str(cross_best_rate) + " and " + str(test_rate) + " in testing.")
    #
    print("_________________________________________________________________________________________________________")
    print("MNIST Dataset")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("mnist_small")

    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=0)

    cross_best_rate, k, distance_type = five_fold_cross(x_merge, y_merge)
    estimate = []
    for j in range(len(x_test)):
        temp = [False]*len(y_test)
        temp[knn_estimate(x_train, y_train, distance_type, x_test[j], 5)] = True
        estimate.append(temp)

    correct = 0
    total = len(y_test)
    for i in range(total):
        for j in range(len(y_test[0])):
            if (y_test[i][j] and estimate[i][j]) or (not y_test[i][j] and not estimate[i][j]):
                correct += 1
                continue

    test_rate = correct / total
    print("The Optimal Configuration is a K-value of " + str(k)+ " and using the " + distance_type + " distance metric")
    print("The success rate at the optimal configuration in cross validation was " + str(cross_best_rate) + " and " + str(test_rate) + " in testing.")
