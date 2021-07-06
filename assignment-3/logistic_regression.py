from data_utils import load_dataset
import numpy, math, random
import matplotlib.pyplot as plt


# Useful helper function: calculates gradient for Log Likelihood Sigmoid Bernoulli probability loss function
# for full batch GD
def logistic_gradient(big_x, estimate, actual):
    grad = 0
    for i in range(len(actual)):
        grad += (actual[i]-estimate[i])*big_x[i]

    return grad


# Useful helper function: calculates gradient for Log Likelihood Sigmoid Bernoulli probability loss function
# for SGD
def sgd_logistic_gradient(x, estimate, actual):
    return (actual-estimate)*x


# Calculates  Log Likelihood Sigmoid Bernoulli probability loss function
def likelihood_loss(estimate, actual):
    likelihood = 0
    for i in range(len(actual)):

        val = math.log(max(0.00001, estimate[i]))
        likelihood += actual[i]*val
        likelihood += (1-actual[i])*math.log(max(0.00001, 1-estimate[i]))
    return likelihood/len(actual)


# Calculates testing accuracy for a given estimate and result vector
def accuracy(estimate, actual):
    count = 0
    for i in range(len(actual)):
        if (actual[i] == 1 and estimate[i] >= 0.5) or (actual[i] == 0 and estimate[i]):
            count += 1
    return count/len(actual)


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


# Implements Logistic Log Likelihood Bernoulli Probability full batch gradient descent
def fb_logistic_classification(xtrain, ytrain, learning_rate):
    # Generates X matrix from training data input
    big_x = numpy.empty((1 + len(xtrain[0]), len(xtrain)))
    big_x[0] = numpy.ones((1, len(xtrain)))
    big_x[1:, :] = xtrain.T
    big_x = big_x.T

    w = numpy.zeros((len(xtrain[0]) + 1, 1))
    ll_vec = []
    accuracy_vec = []

    for it in range(len(ytrain)*5):
        # Calculates estimate for given weight vector
        est = numpy.dot(big_x, w)

        # Calculates rate for descent
        rate = learning_rate * logistic_gradient(big_x, est, ytrain)
        rate = rate[..., numpy.newaxis]
        w += rate

        # Calculates and appends loss likelihood and testing accuracy performance metrics
        ll_vec.append(-1*likelihood_loss(est, ytrain))
        accuracy_vec.append(accuracy(est, ytrain))

    return w, ll_vec, accuracy_vec


# Implements Logistic Log Likelihood Bernoulli Probability stochastic gradient descent
def sgd_logistic_classification(xtrain, ytrain, learning_rate):
    # Generates X matrix from training data input
    big_x = numpy.empty((1 + len(xtrain[0]), len(xtrain)))
    big_x[0] = numpy.ones((1, len(xtrain)))
    big_x[1:, :] = xtrain.T
    big_x = big_x.T

    w = numpy.zeros((len(xtrain[0]) + 1, 1))
    ll_vec = []
    accuracy_vec = []

    for it in range(len(ytrain)*5):
        # Generates random integer index for this iteration
        i = random.randint(0, len(big_x) - 1)

        # Calculates estimate with te weight for this iteration
        est = numpy.dot(big_x, w)

        # Calculates rates for gradient descent
        rate = learning_rate * sgd_logistic_gradient(big_x[i], est[i], ytrain[i])
        rate = rate[..., numpy.newaxis]
        w += rate

        # Calculates and appends log likelihood and testing accuracy performance metrics
        ll_vec.append(-1*likelihood_loss(est, ytrain))
        accuracy_vec.append(accuracy(est, ytrain))

    return w, ll_vec, accuracy_vec


if __name__ == '__main__':

    print("_________________________________________________________________________________________________________")
    print("Full Batch  Gradient Descent")
    x_train, x_valid, x_test, train_bools, y_valid, test_bools = load_dataset("iris")

    y_train = []
    for val in train_bools:
        if val[1]:
            y_train.append(1)
        else:
            y_train.append(0)

    y_test = []
    for val in test_bools:
        if val[1]:
            y_test.append(1)
        else:
            y_test.append(0)

    # Calculating optimal weights for benchmarking
    opt_w = optimal_weights(x_train, y_train)

    xtest_m = numpy.empty((1 + len(x_test[0]), len(x_test)))
    xtest_m[0] = numpy.ones((1, len(x_test)))
    xtest_m[1:, :] = x_test.T
    xtest_m = xtest_m.T
    opt_est = numpy.dot(xtest_m, opt_w)
    for i in range(len(opt_est)):
        opt_est[i] = max(0.000001, opt_est[i])
    opt_ll = likelihood_loss(opt_est,y_test)
    opt_accuracy = accuracy(opt_est,y_test)
    ll_ref_vec = []
    accuracy_ref_vec = []

    for i in range(len(x_train)*5):
        ll_ref_vec.append(-opt_ll)
        accuracy_ref_vec.append(opt_accuracy)

    ll_vecs = []
    accuracy_vecs = []
    rates = [0.000001, 0.00001, 0.0001, 0.001]
    for learn_rate in rates:
        w, ll_vec, accuracy_vec = fb_logistic_classification(x_train, y_train, learn_rate)
        ll_vecs.append(ll_vec)
        accuracy_vecs.append(accuracy_vec)


    for i in range(len(rates)):
        plt.plot(accuracy_vecs[i], label="Learning Rate: {}".format(rates[i]))
    plt.plot(accuracy_ref_vec, '--', label="Reference")
    plt.legend(loc='lower right')
    plt.title("Testing Accuracy vs. Learning Rate")
    plt.ylabel("Testing Accuracy")
    plt.xlabel("Iteration")
    plt.show()

    print("_________________________________________________________________________________________________________")
    print("Stochastic Gradient Descent")
    x_train, x_valid, x_test, train_bools, y_valid, test_bools = load_dataset("iris")

    y_train = []
    for val in train_bools:
        if val[1]:
            y_train.append(1)
        else:
            y_train.append(0)

    y_test = []
    for val in test_bools:
        if val[1]:
            y_test.append(1)
        else:
            y_test.append(0)

    # Calculating optimal weights for benchmarking
    opt_w = optimal_weights(x_train, y_train)

    xtest_m = numpy.empty((1 + len(x_test[0]), len(x_test)))
    xtest_m[0] = numpy.ones((1, len(x_test)))
    xtest_m[1:, :] = x_test.T
    xtest_m = xtest_m.T
    opt_est = numpy.dot(xtest_m, opt_w)
    for i in range(len(opt_est)):
        opt_est[i] = max(0.000001, opt_est[i])
    opt_ll = likelihood_loss(opt_est,y_test)
    opt_accuracy = accuracy(opt_est,y_test)
    ll_ref_vec = []
    accuracy_ref_vec = []

    for i in range(len(x_train)*5):
        ll_ref_vec.append(-opt_ll)
        accuracy_ref_vec.append(opt_accuracy)

    ll_vecs = []
    accuracy_vecs = []
    rates = [0.002, 0.005, 0.01, 0.02]
    for learn_rate in rates:
        w, ll_vec, accuracy_vec = sgd_logistic_classification(x_train, y_train, learn_rate)
        ll_vecs.append(ll_vec)
        accuracy_vecs.append(accuracy_vec)

    for i in range(len(rates)):
        plt.plot(ll_vecs[i], label="Learning Rate: {}".format(rates[i]))
    plt.plot(ll_ref_vec, '--', label="Reference")
    plt.legend()
    plt.title("Log Likelihood Error vs. Learning Rate")
    plt.ylabel("Log Likelihood Error")
    plt.xlabel("Iteration")
    plt.show()

