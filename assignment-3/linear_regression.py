from data_utils import load_dataset
import numpy, math, random
import matplotlib.pyplot as plt

# Useful helper function: allow for loop iteration with floating point numbers
def frange(start, stop, step):
     i = start
     while i < stop:
         yield i
         i += step

# Useful helper function: Calculates root mean squared error between two vectors
def rmse(x1, x2):
    result = 0
    for i in range(len(x1)):
        result += (x1[i] - x2[i]) * (x1[i] - x2[i])
    result = result/len(x1)
    return math.sqrt(result)


# Useful helper function: Calculates the L2 Loss function error
def error_calc(est, act):
    sum = 0
    n = len(act)
    for i in range(n):
        sum += math.pow(est[i]-act[i], 2)
    return sum/n


# Useful helper function: Calculates the gradient associated with the L2 Loss function error for full batch GD
def loss_gradient(est, act, x_mat):
    sum = 0
    n = len(act)
    for i in range(n):
        sum += (est[i]-act[i]) * x_mat[i]

    return 2*sum/n


# Useful helper function: Calculates the gradient associated with the L2 Loss function error for SGD
def sgd_loss_gradient(y, y_est, x):

    rate = 2 * (y_est - y) * numpy.insert(x, 0, 1)

    return rate


# Implements a Full Batch Gradient Descent linear model regression to estimate y values of x test
def linear_regression(xtest, xtrain, ytrain, learning_rate):

    # Generates X matrix from training data input
    big_x = numpy.empty((1+len(xtrain[0]), len(xtrain)))
    big_x[0] = numpy.ones((1, len(xtrain)))
    big_x[1:, :] = xtrain.T
    big_x = big_x.T


    w = numpy.zeros((len(xtrain[0])+1, 1))
    error_vec = []
    print(learning_rate)

    for it in range(len(ytrain)):
        # Calculates estimate with latest weights
        est = numpy.dot(big_x, w)

        # Calculates gradient for descent
        rate = learning_rate * loss_gradient(est, ytrain, big_x)
        rate = rate[..., numpy.newaxis]
        w -= rate
        error = error_calc(est,y_train)
        error_vec.append(error)

    # Returns vector of error per iteration results
    return error_vec


# Calculates the optimal weight vector using the assignment 1 linear regression algorithm
def optimal_weights(xtrain, ytrain):
    # Generates X matrix from training data input
    big_x = numpy.empty((1 + len(xtrain[0]), len(xtrain)))
    big_x[0] = numpy.ones((1, len(xtrain)))
    big_x[1:, :] = xtrain.T
    big_x = big_x.T

    # SVD on X matrix
    u, s, vh = numpy.linalg.svd(big_x)

    # Calculates pseudoinverse using the SVD
    s_inv = numpy.zeros((1 + len(xtrain[0]), len(xtrain)))
    for i in range(1 + len(xtrain[0])):
        s_inv[i][i] = s[i] ** -1
    v_t_s_inv = numpy.dot(vh.T, s_inv)
    temp = numpy.dot(v_t_s_inv, u.T)

    # Uses pseudoinverse to calculate weights from training data
    w = numpy.dot(temp, ytrain)
    return w


# Implements a Stochastic Gradient Descent linear model regression to estimate y values of x test
def sgd_linear_regression(xtest, xtrain, ytrain, learning_rate):

    # Generates X matrix from training data input
    big_x = numpy.empty((1 + len(xtrain[0]), len(xtrain)))
    big_x[0] = numpy.ones((1, len(xtrain)))
    big_x[1:, :] = xtrain.T
    big_x = big_x.T

    w = numpy.zeros((len(xtrain[0]) + 1, 1))
    error_vec = []
    print(learning_rate)

    # Main Iteration Loop
    for it in range(len(ytrain)):
        # Assigns pseudorandom index for stochastic gradient calculation
        i = random.randint(0, len(big_x) - 1)

        # Calculates estimate with current weight vectors
        est = numpy.dot(big_x, w)

        # Calculates SGD gradient for descent
        rate = sgd_loss_gradient(ytrain[i], est[i], big_x[i])/len(est)
        rate = rate[..., numpy.newaxis]
        w -= learning_rate*rate

        # Calculates error and appends to error vector
        error = error_calc(est, y_train)
        error_vec.append(error)

    # Returns weight vector and error per iteration results
    return w, error_vec


if __name__ == '__main__':

    print("_________________________________________________________________________________________________________")
    print("Full Batch Gradient Descent")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("pumadyn32nm")

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    # Calculating optimal weights for benchmarking
    opt_w = optimal_weights(x_train,y_train)

    xtest_m = numpy.empty((1 + len(x_test[0]), len(x_test)))
    xtest_m[0] = numpy.ones((1, len(x_test)))
    xtest_m[1:, :] = x_test.T
    xtest_m = xtest_m.T
    opt_est = numpy.dot(xtest_m, opt_w)
    opt_error = error_calc(opt_est,y_test)
    ref_vec = []
    for i in range(len(x_train)):
        ref_vec.append(opt_error)

    error_vecs = []
    rates = [0.0005,0.001,0.01,0.1,0.5]
    for learn_rate in rates:
        error_vecs.append(linear_regression(x_test, x_train, y_train,learn_rate))

    for i in range(len(rates)):
        plt.plot(error_vecs[i], label="Learning Rate: {}".format(rates[i]))
    plt.plot(ref_vec, '--', label="Reference")
    plt.legend(loc='upper right')
    plt.title("Loss Error per Iteration vs. Learning Rate")
    plt.ylabel("Loss Error")
    plt.xlabel("Iteration")
    plt.show()

    print("_________________________________________________________________________________________________________")
    print("Stochastic Gradient Descent")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("pumadyn32nm")

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    # Calculating optimal weights for benchmarking
    opt_w = optimal_weights(x_train,y_train)

    xtest_m = numpy.empty((1 + len(x_test[0]), len(x_test)))
    xtest_m[0] = numpy.ones((1, len(x_test)))
    xtest_m[1:, :] = x_test.T
    xtest_m = xtest_m.T
    opt_est = numpy.dot(xtest_m, opt_w)
    opt_error = error_calc(opt_est,y_test)
    ref_vec = []
    for i in range(len(x_train)):
        ref_vec.append(opt_error)

    error_vecs = []
    rates = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    for learn_rate in rates:
        w, error_vec = sgd_linear_regression(x_test, x_train, y_train,learn_rate)
        error_vecs.append(error_vec)

    for i in range(len(rates)):
        plt.plot(error_vecs[i], label="Learning Rate: {}".format(rates[i]))
    plt.plot(ref_vec, '--', label="Reference")
    plt.legend(loc='upper right')
    plt.title("Loss Error per Iteration vs. Learning Rate")
    plt.ylabel("Loss Error")
    plt.xlabel("Iteration")
    plt.show()

    estimate = numpy.dot(xtest_m, w)
    plt.plot(y_test)
    plt.plot(estimate)
    plt.show()
    print(rmse(y_test, estimate))
