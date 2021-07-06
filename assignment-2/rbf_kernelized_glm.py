from data_utils import load_dataset
import numpy, math
import matplotlib.pyplot as plt


# rbf_kernelized_glm.py: Trains a generalized linear model with a RBF kernel to perform 1D regression

# Calculates Root Mean Square for two input vectors x1 and x2
def rmse(x1, x2):
    result = 0
    for i in range(len(x1)):
        result += (x1[i] - x2[i]) * (x1[i] - x2[i])
    result = result/len(x1)
    return math.sqrt(result)


def rbf_kernel(x,z, theta):
    # This kernelized implementation uses a gaussian radial basis function rather than the previously derived kernel
    r_square = (numpy.sum((x-z)**2, axis=0))

    return numpy.exp(-r_square/theta)


# Implements a linear model regression to estimate y values of x test
def rbf_kernel_glm(xtest, xtrain, ytrain, lambda_v=0, theta_val=0):

    # Building the K Matrix
    k_mat = numpy.zeros((len(xtrain), len(xtrain)))
    for i in range(len(k_mat)):
        for j in range(len(k_mat)):
            k_mat[i][j] = rbf_kernel(xtrain[i], xtrain[j],theta_val)
            if k_mat[j][i] !=0:
                if k_mat[j][i] != k_mat[i][j]:
                    print("ERROR, Matrix not Symmetrical")

    # #Performing Cholesky decomposition
    r_mat = numpy.linalg.cholesky((k_mat+lambda_v*numpy.eye(len(k_mat))))
    r_inv = numpy.linalg.inv(r_mat)
    temp = numpy.dot(r_inv.T,r_inv)
    alpha = numpy.dot(temp,ytrain)

    # # Build test K matrix
    test_k = numpy.empty((len(xtest), len(xtrain)))
    for i in range(len(xtest)):
        k_vec = numpy.ndarray((len(xtrain)))
        for j in range(len(xtrain)):
            k_vec[j] = rbf_kernel(xtest[i], xtrain[j],theta_val)
        test_k[i, :] = k_vec

    kern_estimate = numpy.dot(test_k, alpha)
    return kern_estimate


def rbf_class(xtest, xtrain, ytrain, lambda_v=0, theta_val=0):

    # Building the K Matrix
    k_mat = numpy.zeros((len(xtrain), len(xtrain)))
    for i in range(len(k_mat)):
        for j in range(len(k_mat)):
            k_mat[i][j] = rbf_kernel(xtrain[i], xtrain[j], theta_val)
            if k_mat[j][i] != 0:
                if k_mat[j][i] != k_mat[i][j]:
                    print("ERROR, Matrix not Symmetrical")

    # #Performing Cholesky decomposition
    r_mat = numpy.linalg.cholesky((k_mat + lambda_v * numpy.eye(len(k_mat))))
    r_inv = numpy.linalg.inv(r_mat)
    temp = numpy.dot(r_inv.T, r_inv)
    # alpha_1 = numpy.dot(temp, ytrain[:, 0])
    # alpha_2 = numpy.dot(temp, ytrain[:, 1])
    alpha = numpy.dot(temp,y_train)

    # # Build test K matrix
    test_k = numpy.empty((len(xtest), len(xtrain)))
    for i in range(len(xtest)):
        k_vec = numpy.ndarray((len(xtrain)))
        for j in range(len(xtrain)):
            k_vec[j] = rbf_kernel(xtest[i], xtrain[j], theta_val)
        test_k[i, :] = k_vec

    estim = numpy.argmax(numpy.dot(test_k,alpha), axis=1)

    # Rebuild a boolean based label vector
    est_vec = []
    for i in range(len(estim)):
        row = [False, False, False]
        row[estim[i]] = True
        est_vec.append(row)

    return est_vec


if __name__ == '__main__':

    print("_________________________________________________________________________________________________________")
    print("Mauna Loa Kernel GLM")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("mauna_loa")
    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=None)

    # Parameter sweep
    min_error = 20000
    opt_params = [0, 0]

    for lambda_val in [0.001, 0.01, 0.1, 1]:
        for theta in [0.05, 0.1, 0.5, 1, 2]:
            estimate = rbf_kernel_glm(x_valid, x_train, y_train, lambda_val, theta)
            error = rmse(estimate, y_valid)
            print("\n New Iteration")
            print(lambda_val)
            print(theta)
            print(error)
            if error < min_error:
                opt_params = [lambda_val, theta]
                min_error = error
            # plt.figure()
            # plt.plot(x_valid, y_valid, label="Testing Labels")
            # plt.plot(x_valid, estimate, label="Estimated Labels")
            # plt.show()


    print("With an optimal regression parameter %f & optimal RBF paremter %f, the RMSE is %f" % (opt_params[0], opt_params[1], min_error))

    estimate = rbf_kernel_glm(x_test, x_train, y_train, opt_params[0], opt_params[1])

    plt.figure()
    plt.plot(x_test, y_test, label="Testing Labels")
    plt.plot(x_test, estimate, label="Estimated Labels")
    plt.xlabel("Testing Feature States")
    plt.ylabel("Label Values")
    plt.title("Kernelized GLM Regression Plot")
    plt.legend(loc='upper left')
    plt.show()

    print("_________________________________________________________________________________________________________")
    print("Rosenbrock RBF GLM")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("rosenbrock", n_train=1000, d=2)
    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=None)

    # Parameter sweep
    min_error = 20000
    opt_params = [0, 0]

    for lambda_val in [0.001, 0.01, 0.1, 1]:
        for theta in [0.05, 0.1, 0.5, 1, 2]:
            estimate = rbf_kernel_glm(x_valid, x_train, y_train, lambda_val, theta)
            error = rmse(estimate, y_valid)
            print("\n New Iteration")
            print(lambda_val)
            print(theta)
            print(error)
            if error < min_error:
                opt_params = [lambda_val, theta]
                min_error = error
            # plt.figure()
            # plt.plot(x_valid, y_valid, label="Testing Labels")
            # plt.plot(x_valid, estimate, label="Estimated Labels")
            # plt.show()

    print("With an optimal regression parameter %f & optimal RBF parameter %f, the RMSE is %f" % (
    opt_params[0], opt_params[1], min_error))

    print("_________________________________________________________________________________________________________")
    print("Iris Dataset")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("iris")

    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=0)

    # Parameter sweep
    max_rate = 0
    opt_params = [0, 0]

    for lambda_val in [0.001, 0.01, 0.1, 1]:
        for theta in [0.05, 0.1, 0.5, 1, 2]:
            estim = rbf_class(x_valid,x_train,y_train,lambda_val,theta)
            correct = 0
            total = len(y_valid)
            for j in range(total):
                cond = False
                for l in range(len(y_valid[j])):
                    if y_valid[j][l] and estim[j][l]:
                        cond = True
                if cond:
                    correct += 1
            rate = correct / total
            print(rate)
            if rate > max_rate:
                max_rate = rate
                opt_params = [lambda_val, theta]

    estim = rbf_class(x_test, x_train, y_train, opt_params[0], opt_params[1])
    correct = 0
    total = len(y_test)
    for j in range(total):
        cond = False
        for l in range(len(y_test[j])):
            if y_test[j][l] and estim[j][l]:
                cond = True
        if cond:
            correct += 1
    rate = correct / total


    print("With an optimal regression parameter %f & optimal RBF parameter %f, the classification rate is %f" % (
        opt_params[0], opt_params[1], max_rate))

