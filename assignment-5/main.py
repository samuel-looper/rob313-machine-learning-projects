from data_utils import load_dataset
import numpy as np
import math
import scipy.stats
from matplotlib import pyplot as plt


# Some useful helper functions

# Calculates classification accuracy between 2 sets
def accuracy(estimate, actual):
    count = 0
    for i in range(len(actual)):
        if (actual[i] == 1 and estimate[i] >= 0.5) or (actual[i] == 0 and estimate[i]):
            count += 1
    return count/len(actual)


# Calculates log likelihood between 2 sets
def log_likelihood(estimate, actual):
    likelihood = 0
    for i in range(len(actual)):
        val = math.log(max(0.00001, estimate[i]))
        likelihood += actual[i]*val
        likelihood += (1-actual[i])*math.log(max(0.00001, 1-estimate[i]))
    return likelihood


# Calculates the gradient for MAP stochastic gradient descent
def map_gradient(x_train, y_train, est, variance, w):
    val = w*(-variance)
    for i in range(len(y_train)):
        temp = (y_train[i] - est[i])[0]
        val += temp * (x_train[i])[..., np.newaxis]
    return val


# Calculates the hessian for Laplace Approximation
def hessian(estimate, x, variance):
    mat = -1/variance*np.identity(len(x[0]))
    for i in range(len(estimate)):
        xvec = (x[i])[..., np.newaxis]
        mat += estimate[i]*(estimate[i]-1)*np.dot(xvec, xvec.T)

    return mat


# Calculates the log prior for Laplace Approximation
def log_prior(w, variance, mean):
    d = len(w)
    val = - 0.5 * (d + 1) * math.log(2 * math.pi * variance)
    for i in range(d):
        val -= (w[i]-mean[i])**2/(2*variance)

    return val


# Calculates the log posterior probability for Laplace Approximation
def log_prob(y, est):
    val = 0
    for i in range(len(y)):

        val += y[i]*math.log(est[i]) + (1-y[i])*math.log(max(0.00001, 1-est[i]))
    return val


# Performs Stochastic Gradient Descent to find Maximum A Posteriori Estimate
def find_map(big_x, w, y_train, variance, learning_rate):
    # calculate MAP using SGD
    for it in range(len(y_train) * 5):
        # Calculates estimate for given weight vector
        est = np.dot(big_x, w)
        est = 1 / (1 + np.exp(-est))
        # Calculates rate for descent
        rate = learning_rate * map_gradient(big_x, y_train, est, variance, w)
        w += rate

        ll = log_likelihood(est, y_train)

    print("MAP found:")
    print(w)
    print("Log Likelihood: %f" % ll)

    return w


# Calculates the bayesian posterior probability
def posterior(x, y, w):
    est = np.dot(x, w)
    est = 1 / (1 + np.exp(-est))
    likelihood = np.exp(log_prob(y, est))
    prior = np.exp(log_prior(w, 1, np.zeros(y.size)))
    return likelihood * prior


if __name__ == '__main__':

    print("_________________________________________________________________________________________________________")
    print("Bayesian Inference Using Laplace Approximation at Different Variances")
    # Load Datasets
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("iris")
    x_train, x_test = np.vstack((x_train, x_valid)), x_test
    y_train, y_test = np.vstack((y_train[:, (1,)], y_valid[:, (1,)])), y_test[:, (1,)]

    # Iterating over 3 different variances
    for variance in [0.5, 1, 2]:
        print("Variance: %f" % variance)

        # Initialize variables for SGD
        big_x = np.empty((1 + len(x_train[0]), len(x_train)))
        big_x[0] = np.ones((1, len(x_train)))
        big_x[1:, :] = x_train.T
        big_x = big_x.T
        w = np.zeros((len(x_train[0]) + 1, 1))
        learning_rate = 0.01

        # Calculate Maximum A Posteriori Estimate
        map_est = find_map(big_x, w, y_train, variance, learning_rate)

        # Calculating an estimate at MAP and Hesian at MAP
        est = np.dot(big_x, map_est)
        est = 1 / (1 + np.exp(-est))
        H = hessian(est, big_x, variance)

        # Calculate Laplace Approximation
        g = -len(map_est)*0.5*math.log(2*math.pi)+0.5*math.log(np.linalg.det(-H))
        log_ml = g + log_prior(map_est, variance, np.zeros(map_est.size)) + log_prob(y_train, est)
        print("Model Log Marginal Likelihood: %f" % log_ml)

    print("_________________________________________________________________________________________________________")
    print("Importance Sampling with Chosen Proposal Distribution")
    # Load datasets
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("iris")
    y_train, y_test, y_valid = y_train[:, (1,)], y_test[:, (1,)], y_valid[:, (1,)]

    map_est = [-0.91615589, 0.26939185, -1.26916876, 0.99227753, -1.19090109]

    # Initialize values for cross-validation
    best_ss = None
    best_var = None
    max_loss = -1000
    big_x_valid = np.empty((1 + len(x_valid[0]), len(x_valid)))
    big_x_valid[0] = np.ones((1, len(x_valid)))
    big_x_valid[1:, :] = x_valid.T
    big_x_valid = big_x_valid.T

    big_x_train = np.empty((1 + len(x_train[0]), len(x_train)))
    big_x_train[0] = np.ones((1, len(x_train)))
    big_x_train[1:, :] = x_train.T
    big_x_train = big_x_train.T

    big_x_test = np.empty((1 + len(x_test[0]), len(x_test)))
    big_x_test[0] = np.ones((1, len(x_test)))
    big_x_test[1:, :] = x_test.T
    big_x_test = big_x_test.T

    # Iterate over 2 hyperparameters: sample size and variance
    for sample_size in [10, 30, 100, 300, 1000]:
        for variance in [1, 2, 5, 10]:
            print(sample_size)
            print(variance)

            # Generate sampling weights from uniform distribution
            big_w = np.empty((sample_size, (len(x_train[0]) + 1)))
            for i in range(sample_size):
                big_w[i] = np.random.multivariate_normal(map_est, variance * np.identity(len(map_est)))

            # For each validation point, generate an estimate using importance sampling
            full_est = []
            for i in range(len(big_x_valid)):
                r_sum = 0
                pred = 0
                # Iterate over every sampled weight in a weighted average according to importance sampling algorithm
                for w in big_w:
                    # Calculate full state estimate with sampled weight
                    est = np.dot(big_x_train, w)
                    est = 1 / (1 + np.exp(-est))

                    # Calculate r value for sample weight and add to weight to average
                    likelihood = np.exp(log_prob(y_train, est))
                    prior = np.exp(log_prior(w, 1, np.zeros(len(map_est))))
                    proposal = np.exp(log_prior(w, variance, map_est))
                    r = likelihood * prior / proposal
                    r_sum += r

                    # Compute estimate for singlel point in the state and add to the prediction sum
                    single_est = np.dot(big_x_valid[i], w)
                    single_est = 1 / (1 + np.exp(-single_est))
                    pred += single_est * r

                # Perform weighted average and add to estimate vector
                pred /= r_sum
                full_est.append(pred)

            # Calculate Log Likelihood and accuracy for validation set to optimize hyperparameters
            log_like = log_likelihood(full_est, np.asarray(y_valid, int))
            print(log_like)
            print("")

            if log_like > max_loss:
                best_ss = sample_size
                best_var = variance
                max_loss = log_like

    # Calculating Test Accuracy & Log Likelihood
    big_w = np.empty((sample_size, len(w)))

    # Generate sampling weights from uniform distribution
    for i in range(sample_size):
        big_w[i] = np.random.multivariate_normal(map_est, variance * np.identity(len(map_est)))

    # For each testing point, generate an estimate using importance sampling
    full_est = []
    for i in range(len(big_x_test)):
        r_sum = 0
        pred = 0

        # Iterate over every sampled weight in a weighted average according to importance sampling algorithm
        for w in big_w:

            # Calculate full state estimate with sampled weight
            est = np.dot(big_x_train, w)
            est = 1 / (1 + np.exp(-est))

            # Calculate r value for sample weight and add to weight to average
            likelihood = np.exp(log_prob(y_train, est))
            prior = np.exp(log_prior(w, 1, np.zeros(len(map_est))))
            proposal = np.exp(log_prior(w, variance, map_est))
            r = likelihood * prior / proposal
            r_sum += r

            # Compute estimate for single point in the state and add to the prediction sum
            single_est = np.dot(big_x_test[i], w)
            single_est = 1 / (1 + np.exp(-single_est))
            pred += single_est * r

        # Perform weighted average and add to estimate vector
        pred /= r_sum
        full_est.append(pred)

    # Calculate Log Likelihood and accuracy
    log_like = log_likelihood(full_est, y_test)
    acc = accuracy(full_est, y_test)

    print("Optimal Sample Size: %f" % best_ss)
    print("Optimal Variance: %f" % best_var)
    print("Test Negative Log Likelihood: %f" % -log_like)
    print("Test Accuracy: %f" % acc)

    # Creating posterior samples to visualize
    sample_weights = np.empty((10000, len(x_train[0])+1))
    for i in range(10000):
        sample_weights[i] = np.random.multivariate_normal(map_est, 2 * np.identity(len(map_est)))

    sample_post = []
    r_sum = 0
    for w in sample_weights:
        est = np.dot(big_x_train, w)
        est = 1 / (1 + np.exp(-est))
        likelihood = np.exp(log_prob(y_train, est))
        prior = np.exp(log_prior(w, 1, np.zeros(len(map_est))))
        proposal = np.exp(log_prior(w, 2, map_est))
        r = likelihood * prior / proposal
        r_sum += r
        sample_post.append(r)

    sample_post /= r_sum

    # Visualizing posterior for each weight index
    for i in range(len(sample_weights[0])):
        weights = sample_weights[:, i]

        w_range = np.arange(min(weights), max(weights), 0.001)
        proposal = scipy.stats.norm.pdf(w_range, map_est[i], 2)
        plt.figure()
        plt.title("1 Dimension Visualization of Proposal vs. Posterior")
        plt.xlabel("Weight vector at index %d" %i)
        plt.ylabel("Probability")
        plt.ylim(top=0.21)
        plt.plot(w_range, proposal, label="Proposal q(w)")
        plt.plot(weights, sample_post, "or", label="Posterior Samples")
        plt.legend()
        plt.show()



    print("_________________________________________________________________________________________________________")
    print("Metropolis Hasting MCMC")

    # Load datasets
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("iris")
    y_train, y_test, y_valid = y_train[:, (1,)], y_test[:, (1,)], y_valid[:, (1,)]
    x_stack = np.vstack((x_train, x_valid))
    y_stack = np.vstack((y_train, y_valid))

    # Initialize values for cross-validation
    big_x_valid = np.empty((1 + len(x_valid[0]), len(x_valid)))
    big_x_valid[0] = np.ones((1, len(x_valid)))
    big_x_valid[1:, :] = x_valid.T
    big_x_valid = big_x_valid.T

    big_x_train = np.empty((1 + len(x_train[0]), len(x_train)))
    big_x_train[0] = np.ones((1, len(x_train)))
    big_x_train[1:, :] = x_train.T
    big_x_train = big_x_train.T

    big_x_test = np.empty((1 + len(x_test[0]), len(x_test)))
    big_x_test[0] = np.ones((1, len(x_test)))
    big_x_test[1:, :] = x_test.T
    big_x_test = big_x_test.T
    w = np.zeros((len(x_stack[0]) + 1, 1))
    learning_rate = 0.01
    max_loss = -1000
    map_est = [-0.91615589, 0.26939185, -1.26916876, 0.99227753, -1.19090109]

    # Iterate over 1 hyperparameters: variance
    for variance in [1, 2, 4, 6, 8, 10]:
        print("Variance: %f" % variance)

        # Initialize parameters for dependent weight sampling
        big_w = []
        means = []
        w_prev = map_est
        w_curr = np.random.multivariate_normal(mean=w_prev, cov=np.eye(len(w_prev)) * variance)

        # Perform Metropolis Hastings sampling for 100 samplings, with 1000 iteration burn in and 100 iteration thinning
        i = 0
        for i in range(11000):

            # Sample new weight from multivariate normal dependent on previous value
            u = np.random.uniform()
            w_new = np.random.multivariate_normal(mean=w_curr, cov=np.eye(len(w_curr)) * variance)
            prob = posterior(big_x_train, y_train, w_new)/ posterior(big_x_train, y_train, w_curr)

            # Uniform distribution condition
            if u < min(1, prob):
                w_prev = w_curr
                w_curr = w_new

            # If statement handles burn-in and thinning
            if i >= 1000 and i % 100 == 0:
                # print(i)
                # print(w_prev)

                # Samples weight and adds to matrix
                j = int((i-1000)/100)
                big_w.append(w_curr)
                means.append(w_prev)

        # For each validation point, generate an estimate using importance sampling
        full_est = []
        for i in range(len(big_x_valid)):
            r_sum = 0
            pred = 0

            # Iterate over every sampled weight in a weighted average according to importance sampling algorithm
            for w in big_w:
                # Calculate full state estimate with sampled weight
                est = np.dot(big_x_train, w)
                est = 1 / (1 + np.exp(-est))

                # Calculate r value for sample weight and add to weight to average
                likelihood = np.exp(log_prob(y_train, est))
                prior = np.exp(log_prior(w, 1, np.zeros(len(map_est))))
                proposal = np.exp(log_prior(w, variance, map_est))
                r = likelihood * prior / proposal
                r_sum += r

                # Compute estimate for singlel point in the state and add to the prediction sum
                single_est = np.dot(big_x_valid[i], w)
                single_est = 1 / (1 + np.exp(-single_est))
                pred += single_est * r

            # Perform weighted average and add to estimate vector
            pred /= r_sum
            full_est.append(pred)

        # Calculate Log Likelihood and accuracy for validation set to optimize hyperparameters
        log_like = log_likelihood(full_est, np.asarray(y_valid, int))
        acc = accuracy(full_est, np.asarray(y_valid, int))
        print("Log Likelihood: %f" % log_like)
        print("Accuracy: %f" % acc)
        print("")

        if log_like > max_loss:
            best_var = variance
            max_loss = log_like

    # MCMC with Testing set for Optimal Parameter Accuracy & Log Likelihood
    variance = 8
    big_w = []
    means = []
    ninth_post = []
    tenth_post = []
    w_prev = map_est
    w_curr = np.random.multivariate_normal(mean=w_prev, cov=np.eye(len(w_prev)) * variance)

    # Perform Metropolis Hastings sampling for 100 samplings, with 1000 iteration burn in and 100 iteration thinning
    i = 0
    for i in range(11000):
        # Sample new weight from multivariate normal dependent on previous value
        u = np.random.uniform()
        w_new = np.random.multivariate_normal(mean=w_curr, cov=np.eye(len(w_curr)) * variance)
        prob = posterior(big_x_train, y_train, w_new) / posterior(big_x_train, y_train, w_curr)

        # Uniform distribution condition
        if u < min(1, prob):
            w_prev = w_curr
            w_curr = w_new

        # If statement handles burn-in and thinning
        if i >= 1000 and i % 100 == 0:
            # print(i)
            # print(w_prev)

            # Samples weight and adds to matrix
            j = int((i - 1000) / 100)
            big_w.append(w_curr)
            means.append(w_prev)

    # For each validation point, generate an estimate using importance sampling
    full_est = []
    for i in range(len(big_x_test)):
        r_sum = 0
        pred = 0

        # Iterate over every sampled weight in a weighted average according to importance sampling algorithm
        for w in big_w:
            # Calculate full state estimate with sampled weight
            est = np.dot(big_x_train, w)
            est = 1 / (1 + np.exp(-est))

            # Calculate r value for sample weight and add to weight to average
            likelihood = np.exp(log_prob(y_train, est))
            prior = np.exp(log_prior(w, 1, np.zeros(len(map_est))))
            proposal = np.exp(log_prior(w, variance, map_est))
            r = likelihood * prior / proposal
            r_sum += r

            # Compute estimate for singlel point in the state and add to the prediction sum
            single_est = np.dot(big_x_test[i], w)
            single_est = 1 / (1 + np.exp(-single_est))
            pred += single_est * r

            # Sampling for single state visualization
            if i == 8:
                ninth_post.append(single_est)
            if i == 9:
                tenth_post.append(single_est)

        # Perform weighted average and add to estimate vector
        pred /= r_sum
        full_est.append(pred)

    # Calculate Test Log Likelihood and Accuracy
    log_like = log_likelihood(full_est, np.asarray(y_test, int))
    acc = accuracy(full_est, np.asarray(y_test, int))
    print("Log Likelihood: %f" % log_like)
    print("Accuracy: %f" % acc)
    print("")

    # Visualizing the 9th and 10th Flower predictive posterior over 100 samples
    print(ninth_post)
    print(tenth_post)
    plt.figure(i)
    plt.title('Predictive Posterior Histogram Flower #9')
    plt.xlabel('P(y*|x*, w(i))')
    plt.xlim((0, 1))
    plt.ylabel('# Occurrences')
    plt.hist(ninth_post, bins=20)
    plt.show()

    plt.figure(i)
    plt.title('Predictive Posterior Histogram Flower #10')
    plt.xlabel('P(y*|x*, w(i))')
    plt.xlim((0, 1))
    plt.ylabel('Count over 100 samples')
    plt.hist(tenth_post, bins=20)
    plt.show()