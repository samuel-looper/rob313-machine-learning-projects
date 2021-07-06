from data_utils import load_dataset
import numpy, math
import matplotlib.pyplot as plt


# kernelized_glm.py: Designs a 1D kernel, trains a kernelized generalized linear model to perform 1D regression

# Calculates Root Mean Square for two input vectors x1 and x2
def rmse(x1, x2):
    result = 0
    for i in range(len(x1)):
        result += (x1[i] - x2[i]) * (x1[i] - x2[i])
    result = result/len(x1)
    return math.sqrt(result)


def kernel(x,z):
    # The kernel is derived by taking the inner product of the phi vector for two arbitrary vectors using the basis
    # functions defined in the first part of the assignment. The (1+xz)^2 term is found by scaling the xz term and
    # and completing the square, whereas the cos term is found by the double angle formula for
    # x&z sin(x&z) and x&z cos(x&z)

    freq = 2 * numpy.pi / 0.055
    return numpy.power((1 + numpy.dot(x, z)), 2) + numpy.dot((1 + numpy.dot(x, z)), numpy.cos(freq*(x-z)))


# Implements a linear model regression to estimate y values of x test
def kernel_glm(xtest, xtrain, ytrain, lambda_val=0):

    # Building the K Matrix
    k_mat = numpy.zeros((len(xtrain), len(xtrain)))
    for i in range(len(k_mat)):
        for j in range(len(k_mat)):
            k_mat[i][j] = kernel(xtrain[i], xtrain[j])
            if k_mat[j][i] !=0:
                if k_mat[j][i] != k_mat[i][j]:
                    print("ERROR, Matrix not Symmetrical")

    # #Performing Cholesky decomposition
    r_mat = numpy.linalg.cholesky((k_mat+lambda_val*numpy.eye(len(k_mat))))
    r_inv = numpy.linalg.inv(r_mat)
    temp = numpy.dot(r_inv.T,r_inv)
    alpha = numpy.dot(temp,ytrain)

    # # Build test K matrix
    test_k = numpy.empty((len(xtest), len(xtrain)))
    for i in range(len(xtest)):
        k_vec = numpy.ndarray((len(xtrain)))
        for j in range(len(xtrain)):
            k_vec[j] = kernel(xtest[i], xtrain[j])
        test_k[i, :] = k_vec

    kern_estimate = numpy.dot(test_k, alpha)
    return kern_estimate


if __name__ == '__main__':

    print("_________________________________________________________________________________________________________")
    print("Mauna Loa Kernel GLM")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("mauna_loa")
    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=None)

    # Lambda value sweep
    min_error = 20000
    opt_lambda = 0

    for lambda_val in range(1,20):
        estimate = kernel_glm(x_valid,x_train,y_train, lambda_val)
        # print(estimate.shape)
        # print(y_valid.shape)
        error = rmse(estimate, y_valid)
        if error<min_error:
            opt_lambda = lambda_val
            min_error = error
        print(lambda_val)

    print("With an optimal value of %d, the RMSE is %f" % (opt_lambda, min_error))

    estimate = kernel_glm(x_test, x_train, y_train, opt_lambda)

    plt.figure()
    plt.plot(x_test, y_test, label="Testing Labels")
    plt.plot(x_test, estimate, label="Estimated Labels")
    plt.xlabel("Testing Feature States")
    plt.ylabel("Label Values")
    plt.title("Kernelized GLM Regression Plot")
    plt.legend(loc='upper left')
    plt.show()

    kernel_vec1 = []
    kernel_vec2 = []
    x_vec = []
    for i in range(100):
        i = -0.1 + 0.2*i/100
        kernel_vec1.append(kernel(0,i))
        kernel_vec2.append(kernel(1,1+i))
        x_vec.append(i)

    plt.figure()
    plt.plot(x_vec, kernel_vec1, label="k(0,z)")
    plt.plot(x_vec, kernel_vec2, label="k(1,z+1)")
    plt.xlabel("Z-value")
    plt.ylabel("Kernel value")
    plt.title("Effects of Translation on Kernel Output")
    plt.legend(loc='upper left')
    plt.show()