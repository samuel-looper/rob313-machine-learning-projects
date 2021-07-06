# ROB313 Machine Learning Projects
Assignments and projects from ROB313 (Intro to Learning with Data). Project code is written in Python. 

Assignment #1: K Nearest Neighbor Algorithm
- classification.py: 		Performs image classification using k Nearest Neighbor algorithm
- regression.py: 		Performs regression using k Nearest Neighbors algorithm, variations evaluated using k-fold cross-validation
- regression_variations.py: 	Evaluates the computational complexity of kNN algorithm variations

Assignment #2: Generalized Linear Models
- non_kernelized_glm.py: 	Trains and evaluates a generalized linear model to perform 1D regression 
- kernelized_glm.py: 		Designs a 1D kernel, trains a kernelized generalized linear model to perform 1D regression 
- rbf_kernelized_glm.py: 	Trains a generalized linear model with a RBF kernel to perform 1D regression

Assignment #3: Stochastic Optimization
- linear_regression.py: 	Performs stochastic gradient descent to optimize parameters in a linear regression model using maximum likelihood, 
				performs hyperparameter optimization
- logisic_regression.py: 	Performs stochastic gradient descent to optimize parameters in a logistic regression model using maximum likelihood

Assignment #4: Artificial Neural Networks
- neural_net.py: 		Implements simple 1-layer feed-forward neural network (from scratch, using Autograd) for image classification. Includes 
				functions to visualize layers. 

Assignment #5: Bayesian Networks
- bayes_net.py: 		Implements variations of Bayesian Networks for binary classification, including models that leverage importance sampling
				and Monte Carlo Markov Chain sampling using the Metropolis-Hastings algorithm. 
