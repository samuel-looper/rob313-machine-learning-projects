import autograd.numpy as np
from autograd import value_and_grad
from data_utils import load_dataset, plot_digit
import matplotlib.pyplot as plt


# Helper function to update parameters based on a gradient and learning rate
def update_parameters(w, grad_w, rate=1.):
    return w - rate * grad_w


# Forward Pass function modified for question 1
def multiclass_forward_pass(W1, W2, W3, b1, b2, b3, x):

    H1 = np.maximum(0, np.dot(x, W1.T) + b1.T)  # layer 1 neurons with ReLU activation, shape (N, M)
    H2 = np.maximum(0, np.dot(H1, W2.T) + b2.T)  # layer 2 neurons with ReLU activation, shape (N, M)
    Fhat = np.dot(H2, W3.T) + b3.T  # layer 3 (output) neurons with linear activation, shape (N, 10)
    # print(Fhat[122])
    # Implements Softmax using the numerically stable LogSumExp method
    a = np.expand_dims(np.amax(Fhat, axis=1), axis=1) # Creates array of max f values

    # Calculates subtraction factor for log of the softmax value
    diff = np.add(a, np.expand_dims(np.log(np.sum(np.exp(np.subtract(Fhat, a)), axis=1)), axis=1))
    log_soft_max = np.subtract(Fhat, diff)      # The Log SoftMax for question 1
    # Calculates final softMax values
    soft_max = np.exp(log_soft_max)
    # print(soft_max[122])
    return soft_max


# Modified Negative Log Likelihood function for question 2
def bernoulli_log_likelihood(W1, W2, W3, b1, b2, b3, x, y):

    # Calculates a class estimate based on a forward pass through the Neural Net
    fhat = multiclass_forward_pass(W1, W2, W3, b1, b2, b3, x)
    # Gaussian Negative Log Likelihood
    # nll = 0.5*np.sum(np.square(Fhat - y)) + 0.5*y.size*np.log(2.*np.pi)

    # Bernoulli Negative Log Likelihood
    nll = -1*np.sum(np.multiply(np.log(fhat), y))

    return nll


def calc_accuracy(W1, W2, W3, b1, b2, b3, x, y):
    # Calculates a class estimate based on a forward pass through the Neural Net
    fhat = multiclass_forward_pass(W1, W2, W3, b1, b2, b3, x)

    correct = 0
    for j in range(len(fhat)):
        guess = None
        actual = None
        max_val = 0
        for i in range(len(fhat[0])):
            if fhat[j][i] > max_val:
                max_val = fhat[j][i]
                guess = i

            if y[j][i]:
                actual = i

        if guess == actual:
            correct +=1

    return correct / len(fhat)


def extract_weird_vals(W1, W2, W3, b1, b2, b3, x, y):
    # Calculates a class estimate based on a forward pass through the Neural Net
    fhat = multiclass_forward_pass(W1, W2, W3, b1, b2, b3, x)
    weird_vals=[]
    for j in range(len(fhat)):
        guess = None
        actual = None
        max_val = 0
        for i in range(len(fhat[0])):
            if fhat[j][i] > max_val:
                max_val = fhat[j][i]
                guess = i

            if y[j][i]:
                actual = i

        if guess != actual and max_val < 0.35:
            print(max_val)
            weird_vals.append(x[j])
    return weird_vals


if __name__ == "__main__":
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    x_train, y_train = x_train[:250], y_train[:250] # Setting up Mini Batch

    # QUESTION #3: 3 LAYER NEURAL NET
    # -----------------------------------------------------------------------------------------------------------------
    # # Initializing Neural Net
    M = 100  # 100 neurons per hidden layer
    W1 = np.random.randn(M, 784)/np.sqrt(784)  # weights of first (hidden) layer
    W2 = np.random.randn(M, M)/np.sqrt(784)  # weights of second (hidden) layer
    W3 = np.random.randn(10, M)/np.sqrt(784)  # weights of third (output) layer
    b1 = np.zeros((M, 1))  # biases of first (hidden) layer
    b2 = np.zeros((M, 1))  # biases of second (hidden) layer
    b3 = np.zeros((10, 1))  # biases of third (output) layer
    nll_gradients = value_and_grad(bernoulli_log_likelihood, argnum=[0, 1, 2, 3, 4, 5])
    learning_rate = 0.0002
    valid_errors = []
    train_errors = []

    # Calculates an estimate based on 2000 iterations of optimization on the Neural Net for the training set
    for i in range(200):
        # compute the gradient
        (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = nll_gradients(W1, W2, W3, b1, b2, b3, x_train, y_train)
        # update the parameters
        W1 = update_parameters(W1, W1_grad, learning_rate)
        W2 = update_parameters(W2, W2_grad, learning_rate)
        W3 = update_parameters(W3, W3_grad, learning_rate)
        b1 = update_parameters(b1, b1_grad, learning_rate)
        b2 = update_parameters(b2, b2_grad, learning_rate)
        b3 = update_parameters(b3, b3_grad, learning_rate)
        error = bernoulli_log_likelihood(W1, W2, W3, b1, b2, b3, x_train, y_train)
        train_errors.append(error)
        # print loss if necessary
        if i == 0 or (i + 1) % 200 == 0:
            print("Iter %3d, loss = %.6f" % (i + 1, error))

    # Reinitialize all the weight values by Xavier Initialization for the validation set estimate
    W1 = np.random.randn(M, 784) / np.sqrt(784)  # weights of first (hidden) layer
    W2 = np.random.randn(M, M) / np.sqrt(784)  # weights of second (hidden) layer
    W3 = np.random.randn(10, M) / np.sqrt(784)  # weights of third (output) layer
    b1 = np.zeros((M, 1))  # biases of first (hidden) layer
    b2 = np.zeros((M, 1))  # biases of second (hidden) layer
    b3 = np.zeros((10, 1))  # biases of third (output) layer

    for i in range(200):
        # compute the gradient
        (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = nll_gradients(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        # update the parameters
        W1 = update_parameters(W1, W1_grad, learning_rate)
        W2 = update_parameters(W2, W2_grad, learning_rate)
        W3 = update_parameters(W3, W3_grad, learning_rate)
        b1 = update_parameters(b1, b1_grad, learning_rate)
        b2 = update_parameters(b2, b2_grad, learning_rate)
        b3 = update_parameters(b3, b3_grad, learning_rate)
        error = bernoulli_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        valid_errors.append(error)
        # print loss if necessary
        if i == 0 or (i + 1) % 200 == 0:
            print("Iter %3d, loss = %.6f" % (i + 1, error))

    # Plot values for part 3
    plt.plot(train_errors, label="Training Log Likelihood")
    plt.plot(valid_errors, label="Validation Log Likelihood")
    plt.legend()
    plt.title("Log Likelihood Error over 2000 Iterations")
    plt.ylabel("Log Likelihood Error")
    plt.xlabel("Iteration")
    plt.show()

    # QUESTION #4: VARYING NUMBER OF NEURONS FOR 3-LAYER NEURAL NET
    # -----------------------------------------------------------------------------------------------------------------

    for M in [50,100,150]:
        print("{} Neurons per Layer".format(M))

        W1 = np.random.randn(M, 784) / np.sqrt(784)  # weights of first (hidden) layer
        W2 = np.random.randn(M, M) / np.sqrt(784)  # weights of second (hidden) layer
        W3 = np.random.randn(10, M) / np.sqrt(784)  # weights of third (output) layer
        b1 = np.zeros((M, 1))  # biases of first (hidden) layer
        b2 = np.zeros((M, 1))  # biases of second (hidden) layer
        b3 = np.zeros((10, 1))  # biases of third (output) layer
        nll_gradients = value_and_grad(bernoulli_log_likelihood, argnum=[0, 1, 2, 3, 4, 5])
        learning_rate = 0.0002
        valid_errors = []
        train_errors = []

        for i in range(200):
            # compute the gradient
            (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = nll_gradients(W1, W2, W3, b1, b2, b3, x_train, y_train)
            # update the parameters
            W1 = update_parameters(W1, W1_grad, learning_rate)
            W2 = update_parameters(W2, W2_grad, learning_rate)
            W3 = update_parameters(W3, W3_grad, learning_rate)
            b1 = update_parameters(b1, b1_grad, learning_rate)
            b2 = update_parameters(b2, b2_grad, learning_rate)
            b3 = update_parameters(b3, b3_grad, learning_rate)

        print("Validation Set")
        error = bernoulli_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        print("Log Likelihood Loss = %.6f" % error)
        accuracy = calc_accuracy(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        print("Accuracy = %.6f" % accuracy)

        print("Testing Set")
        error = bernoulli_log_likelihood(W1, W2, W3, b1, b2, b3, x_test, y_test)
        print("Log Likelihood Loss = %.6f" % error)
        accuracy = calc_accuracy(W1, W2, W3, b1, b2, b3, x_test, y_test)
        print("Accuracy = %.6f" % accuracy)

    # QUESTION #5: Visualizing weights from the Neural Net (Note: Run with Part 4)
    # -----------------------------------------------------------------------------------------------------------------

    # randomly selects 26 of the first layer weights and plots them using the plot_digit helper function
    for it in range(16):
        i = np.random.randint(0,M)
        plot_digit(W1[i])
        plt.show()

    # QUESTION #5: Visualizing problematic inputs (Note: Run with Part 4)
    # -----------------------------------------------------------------------------------------------------------------

    # Finds problematic input values using the helper function from the testing sets.
    weird_vals = extract_weird_vals(W1, W2, W3, b1, b2, b3, x_test, y_test)

    # Plots each of the problematic values using the plot_digit helper function
    for val in weird_vals:
        plot_digit(val)
        plt.show()

