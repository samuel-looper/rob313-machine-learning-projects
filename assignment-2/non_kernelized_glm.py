from data_utils import load_dataset
import numpy, math
import matplotlib.pyplot as plt


# non_kernelized_glm.py: Trains and evaluates a generalized linear model to perform 1D regression

# Calculates Root Mean Square for two input vectors x1 and x2
def rmse(x1, x2):
    result = 0
    for i in range(len(x1)):
        result += (x1[i] - x2[i]) * (x1[i] - x2[i])
    result = result/len(x1)
    return math.sqrt(result)


# Implements a linear model regression to estimate y values of x test
def mauna_loa_glm(xvalid, xtrain, ytrain, lambda_val=0):

    # Generate basis function vectors and matrix for training and test set
    freq = 2 * numpy.pi / 0.055

    # Generate basis function vectors and matrix & generates X matrix from testing data input
    # We choose a truncated polynomial basis for our as the Mauna Loa data looks like it can be approximated by a
    # function of form x sin(x) which can be pretty well estimated by a taylor series which is comprised of a simple
    # series of polynomials with various coefficients

    big_phi = numpy.hstack([numpy.ones((len(xtrain), 1)), xtrain])
    big_phi = numpy.hstack([big_phi, numpy.power(xtrain, 2)])
    big_phi = numpy.hstack([big_phi, xtrain * numpy.sin(freq * xtrain)])
    big_phi = numpy.hstack([big_phi, xtrain * numpy.cos(freq * xtrain)])
    big_phi = numpy.hstack([big_phi, numpy.sin(freq * xtrain)])
    big_phi = numpy.hstack([big_phi, numpy.cos(freq * xtrain)])

    phi_test = numpy.hstack([numpy.ones((len(xvalid), 1)), xvalid])
    phi_test = numpy.hstack([phi_test, numpy.power(xvalid, 2)])
    phi_test = numpy.hstack([phi_test, xvalid * numpy.sin(freq * xvalid)])
    phi_test = numpy.hstack([phi_test, xvalid * numpy.cos(freq * xvalid)])
    phi_test = numpy.hstack([phi_test, numpy.sin(freq * xvalid)])
    phi_test = numpy.hstack([phi_test, numpy.cos(freq * xvalid)])


    # SVD on Phi matrix
    u, s, vh = numpy.linalg.svd(big_phi, full_matrices=True)

    # Calculates pseudoinverse using the SVD
    big_s = numpy.zeros((len(xtrain),len(s)))
    for i in range(len(s)):
        big_s[i][i] = s[i]

    temp = numpy.dot(big_s.T, big_s)
    temp = numpy.linalg.pinv(temp+lambda_val*numpy.eye(len(s)))
    temp = numpy.dot(vh.T, temp)
    temp = numpy.dot(temp, big_s.T)
    temp = numpy.dot(temp, u.T)

    # Uses pseudoinverse to calculate weights from training data
    w = numpy.dot(temp, ytrain)

    # Returns estimate by multiplying testing data by the weights
    return numpy.dot(phi_test, w)


if __name__ == '__main__':
    print("_________________________________________________________________________________________________________")
    print("Mauna Loa GLM")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("mauna_loa")
    x_merge = numpy.concatenate((x_valid, x_train), axis=0)
    y_merge = numpy.concatenate((y_valid, y_train), axis=None)

    # Lambda value sweep
    min_error = 20000
    opt_lambda = 0

    for lambda_val in range(20):
        estimate = mauna_loa_glm(x_valid,x_train,y_train, lambda_val)
        error = rmse(estimate, y_valid)
        if error<min_error:
            opt_lambda = lambda_val
            min_error = error

    print("With an optimal value of %d, the RMSE is %f" % (opt_lambda, min_error))

    estimate = mauna_loa_glm(x_test, x_train, y_train, opt_lambda)

    plt.figure()
    plt.plot(x_test, y_test, label="Testing Labels")
    plt.plot(x_test, estimate, label="Estimated Labels")
    plt.xlabel("Testing Feature States")
    plt.ylabel("Label Values")
    plt.title("Non Kernelized GLM Regression Plot")
    plt.legend(loc='upper left')
    plt.show()
