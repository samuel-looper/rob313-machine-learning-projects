from data_utils import load_dataset
import math
import numpy
import matplotlib.pyplot as plt


# regression.py: Performs regression using k Nearest Neighbors algorithm, variations evaluated using k-fold cross-validation

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


# Returns the indices of the k closest neighbors in "xarray" for the x value "centerpoint"
def get_neighbors(distancetype, xarray, centerpoint, k):
    distances = []

    # Pre-processes the inputs to ensure the centerpoint is always represented as a 1x1 error (patch for errors in debug)
    if isinstance(centerpoint,list):
        centerpoint = centerpoint[0]
    elif isinstance(centerpoint,numpy.float64):
        centerpoint = [centerpoint]

    for i in range(len(xarray)):
        # Pre-processes the inputs to ensure all values of xarray are always represented as a 1x1 error (patch for errors in debug)
        if isinstance(xarray[i], numpy.float64):
            arr = [xarray[i]]
        else:
            arr = xarray[i]
        # Creates a distance array, with the distance from every point in xarray to the centerpoint, and the xarray values' orignal index
        temp = distance(distancetype, centerpoint, arr)
        distances.append((temp, i))

    # Sorts the distance array with respect to the distance & creates neighbors array with the indices of the k closest values
    distances.sort(key=lambda tup: tup[0])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][1])

    return neighbors


# Returns the average of the k closest neighbors y values for a given centerpoint
def knn_estimate (xarray, yarray, distanceType, centerpoint, k):

    neighbors = get_neighbors(distanceType, xarray, centerpoint, k)
    tot = 0
    for ind in neighbors:
        tot += yarray[ind]
    result = tot/k

    return result


# Performs 5 fold Cross Validation for the Regression kNN, returning the optimal k value, distance metric, and the average RMSE error
def five_fold_cross(x, y):

    # Shuffles the x and y array (ensuring corresponding values have the same index
    merged = []
    for i in range(len(x)):
        merged.append([y[i], x[i][0]])

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

    # Iterates through every distance metric, k values from 1 to 9, and performs a 5 fold cross validation for each configuration
    min_rmse_error = math.inf
    result = [None, None, None]
    error_array = []
    for dist_type in ["L1","L2","L00"]:
        for k_val in range(1, 10):
            rmse_error = 0
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

                #Builds an array of estimates using the kNN algorithm
                estim = []
                for j in range(len(x_testing)):
                    estim.append(knn_estimate(x_training, y_training, dist_type, x_testing[j], k_val))

                #Calculates RMSE, which is averaged over all 5 folds
                temp = rmse(estim,y_testing)
                rmse_error += temp


            rmse_error = rmse_error/5
            error_array.append(rmse_error)
            if rmse_error < min_rmse_error:
                min_rmse_error = rmse_error
                result = [error_array,min_rmse_error,k_val,dist_type]

    return result[0], result[1], result[2], result[3]



if __name__ == '__main__':
    print("_________________________________________________________________________________________________________")
    print("Mauna Loa Dataset")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("mauna_loa")

    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=None)

    error_array,cross_rmse, k, distance_type = five_fold_cross(x_merge, y_merge)

    estimate = []
    for j in range(len(x_test)):
        estimate.append(knn_estimate(x_test, y_test, distance_type, [x_test[j]], k))

    test_rmse = rmse(estimate, y_test)


    print("The Optimal Configuration is a K-value of " + str(k)+ " and using the " + distance_type + " distance metric")
    print("The RMSE at the optimal configuration was " + str(test_rmse) + " in testing and " + str(cross_rmse) + " in cross-validation")
    plt.plot(error_array)
    plt.ylabel('RMSE Error')
    plt.xlabel('K-value')
    plt.axes([0, 20, 0, 0.5])

    plt.show()

    print("_________________________________________________________________________________________________________")
    print("Rosenbrock Dataset")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("rosenbrock", n_train = 1000, d=2)

    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=None)

    cross_rmse, k, distance_type = five_fold_cross(x_merge, y_merge)

    estimate = []
    for j in range(len(x_test)):
        estimate.append(knn_estimate(x_merge, y_merge, distance_type, [x_test[j]], k))

    test_rmse = rmse(estimate, y_test)

    print(
        "The Optimal Configuration is a K-value of " + str(k) + " and using the " + distance_type + " distance metric")
    print("The RMSE at the optimal configuration was " + str(test_rmse) + " in testing and " + str(cross_rmse) + " in cross-validation")
    plt.plot(y_valid)
    plt.plot(estimate)
    plt.show()

    print("_________________________________________________________________________________________________________")
    print("Puma 560 Dataset")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("pumadyn32nm")
    print("Data Loaded")

    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=None)

    #cross_rmse, k, distance_type = five_fold_cross(x_merge, y_merge)
    cross_rmse, k, distance_type = None, 9, "L1"
    estimate = []
    for j in range(len(x_test)):
        estimate.append(knn_estimate(x_merge, y_merge, distance_type, [x_test[j]], k))

    test_rmse = rmse(estimate, y_test)


    print(
        "The Optimal Configuration is a K-value of " + str(k) + " and using the " + distance_type + " distance metric")
    print("The RMSE at the optimal configuration was " + str(test_rmse) + " in testing and " + str(
        cross_rmse) + " in cross-validation")
    plt.plot(y_test)
    plt.plot(estimate)
    plt.show()
