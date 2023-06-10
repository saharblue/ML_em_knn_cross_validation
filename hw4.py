import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def _hypothesis(self, x, theta):
        return 1 / (1 + np.exp(-x.dot(theta)))

    def _cost(self, X, y, theta):
        hypo = self._hypothesis(X, theta)
        return (-y * np.log(hypo) - (1 - y) * np.log(1 - hypo)).sum() / (X.shape[0])

    def _apply_bias_trick(self, X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        self.thetas = list()
        self.Js = list()
        X = self._apply_bias_trick(X)
        self.theta = np.zeros(X.shape[1]) #np.random.random(size=(X.shape[1]))
        prev_cost = 0

        for _ in range(self.n_iter):
            gradient = (self._hypothesis(X, self.theta) - y).dot(X)
            self.theta -= self.eta * gradient
            self.thetas.append(self.theta)
            new_cost = self._cost(X, y, self.theta)
            self.Js.append(new_cost)
            if abs(prev_cost - new_cost) < self.eps:
                break
            prev_cost = new_cost

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None

        X = self._apply_bias_trick(X)
        preds = 1 * (self._hypothesis(X, self.theta) > 0.5)

        return preds


def cross_validation(X, y, folds, algo: LogisticRegressionGD, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    param_values = np.hstack((X, y[:, np.newaxis]))
    np.random.shuffle(param_values)
    chunks = np.array_split(param_values, folds, axis=0)
    accurecies = list()
    for chunk in chunks:
        test_data = chunk
        train_data = None
        for c in chunks:
            if c is chunk:
                continue
            if train_data is None:
                train_data = c
            else:
                train_data = np.vstack((train_data, c))

        algo.fit(train_data[:, :-1], train_data[:, -1])
        pred = algo.predict(test_data[:, :-1])
        accuracy = test_data[test_data[:, -1] == pred].shape[0] / test_data.shape[0]
        accurecies.append(accuracy)

    cv_accuracy = np.array(accurecies).mean()

    return cv_accuracy

def norm_pdf(x, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    p = np.exp(-(np.square((x - mu) / (sigma)))/2) / (sigma * np.sqrt(2 * math.pi))
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = list()

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        # self.weights = np.full(self.k, (1 / self.k))
        # self.mus = np.random.random(self.k)
        # self.sigmas = np.random.random(self.k)
        np.random.seed(self.random_state)
        num_samples, num_features = data.shape

        # pick k random indexes to be the initial mus
        index = np.random.choice(num_samples, self.k, replace=False)
        self.mus = data[index][:, 0]
        # initialize all the weights to be 1 / k
        self.weights = np.full(self.k, 1 / self.k)
        # initialize all the sigmas to be the std of the data
        # #self.sigmas = np.full((self.k, num_features), data.std(axis=0))
        self.sigmas = np.ones((self.k,)) * np.std(data)

    def calculate_pdf_array(self, data):
        pdf_array = []
        for i in range(self.k):
            pdf_array.append(norm_pdf(data, self.mus[i], self.sigmas[i]))
        return pdf_array

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        norm_values = norm_pdf(data, self.mus, self.sigmas) * self.weights
        denominator = norm_values.sum(axis=1)
        self.responsibilities = norm_values / denominator[:, np.newaxis]

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.weights = self.responsibilities.sum(axis=0) / data.shape[0]
        self.mus = (self.responsibilities * data).sum(axis=0) / (data.shape[0] * self.weights)
        self.sigmas = np.sqrt((self.responsibilities * np.square(data - self.mus)).sum(axis=0) / (data.shape[0] * self.weights))

    def cost(self, data):
        return -(np.log2(norm_pdf(data, self.mus, self.sigmas) * self.weights).sum(axis=1)).sum()

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        for _ in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            self.costs.append(self.cost(data))
            if len(self.costs) > 1:
                if abs(self.costs[-1] - self.costs[-2]) < self.eps:
                    break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

# class EM(object):
#     """
#     Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.
#
#     Parameters
#     ------------
#     k : int
#       Number of gaussians in each dimension
#     n_iter : int
#       Passes over the training dataset in the EM proccess
#     eps: float
#       minimal change in the cost to declare convergence
#     random_state : int
#       Random number generator seed for random params initialization.
#     """
#
#     def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
#         self.k = k
#         self.n_iter = n_iter
#         self.eps = eps
#         self.random_state = random_state
#
#         np.random.seed(self.random_state)
#
#         self.responsibilities = []
#         self.weights = []
#         self.mus = []
#         self.sigmas = []
#         self.costs = []
#
#     # initial guesses for parameters
#     def init_params(self, data):
#         """
#         Initialize distribution params
#         """
#         np.random.seed(self.random_state)
#         num_samples, num_features = data.shape
#
#         # pick k random indexes to be the initial mus
#         index = np.random.choice(num_samples, self.k, replace=False)
#         self.mus = data[index]
#         # initialize all the weights to be 1 / k
#         self.weights = np.full(self.k, 1 / self.k)
#         # initialize all the sigmas to be the std of the data
#         # #self.sigmas = np.full((self.k, num_features), data.std(axis=0))
#         self.sigmas = np.ones((self.k,)) * np.std(data)
#
#     def calculate_pdf_array(self, data):
#         pdf_array = []
#         for i in range(self.k):
#             pdf_array.append(norm_pdf(data, self.mus[i], self.sigmas[i]))
#         return pdf_array
#
#     def expectation(self, data):
#         """
#         E step - This function should calculate and update the responsibilities
#         """
#         self.responsibilities = np.zeros((self.k, int(data.shape[0])))
#         for i in range(self.k):
#             for j, row in enumerate(data):
#                 total_pdf = np.array(self.calculate_pdf_array(row))
#                 self.responsibilities[i, j] = (self.weights[i] * norm_pdf(row, self.mus[i], self.sigmas[i]).flatten()) / \
#                                               sum(np.array(self.weights).dot(total_pdf))
#
#     def maximization(self, data):
#         """
#         M step - This function should calculate and update the distribution params
#         """
#         num_samples = data.shape[0]
#         # update mus
#         for i in range(self.k):
#             # update weights
#             self.weights[i] = sum(self.responsibilities[i]) / num_samples
#             # update mus
#             self.mus[i] = sum(self.responsibilities[i].dot(data)) / (self.weights[i] * num_samples)
#             # update sigmas
#             self.sigmas[i] = np.sqrt(
#                 sum(np.array(self.responsibilities[i]).dot((data - self.mus[i]) ** 2)) /
#                 (self.weights[i] * num_samples))
#
#     def log_likelihood(self, data):
#         """
#         Calculates the loglikelihood of the data given the current parameters.
#
#         Parameters
#         ----------
#         data: the data we want to calculate log likelihood for.
#
#         Returns
#         -------
#         loglikelihood of the data given the current parameters.
#         """
#         sum_all = 0
#         for i in range(data.shape[0]):
#             for j in range(self.k):
#                 internal_sum = 0
#                 internal_sum += self.weights[j] * norm_pdf(data[i], self.mus[j], self.sigmas[j])
#             sum_all += np.log(internal_sum)
#         return sum_all
#
#     def fit(self, data):
#         """
#         Fit training data (the learning phase).
#         Use init_params and then expectation and maximization function in order to find params
#         for the distribution.
#         Store the params in attributes of the EM object.
#         Stop the function when the difference between the previous cost and the current is less than eps
#         or when you reach n_iter.
#         """
#         self.init_params(data)
#         last_cost = 0
#         import sys
#         new_cost = sys.maxsize
#         for i in range(self.n_iter):
#             self.expectation(data)
#             self.maximization(data)
#             last_cost = new_cost
#             new_cost = self.log_likelihood(data)
#             if abs(new_cost - last_cost) < self.eps:
#                 break
#
#     def get_dist_params(self):
#         return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    pdf = np.array([norm_pdf(data, mus[i], sigmas[i]) for i in range(mus.shape[0])]).T * weights
    # pdf = (np.apply_along_axis(lambda i: norm_pdf(data, mus[i], sigmas[i]), 0, range(mus.shape[0])) * weights).sum(axis=0)
    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.classes = list()
        self.probability_parameters = dict()

    def init_prior(self, y):
        self.classes, self.prior = np.unique(y, return_counts=True)
        self.classes = self.classes.tolist()
        self.prior = self.prior / y.shape[0]
        self.probability_parameters = {k: list() for k in self.classes}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        self.init_prior(y)
        for label in self.classes:
            for feature in range(X.shape[1]):
                em_object = EM(k=self.k, random_state=self.random_state)
                em_object.fit(X[y == label, feature][:, np.newaxis])
                self.probability_parameters[label].append(em_object.get_dist_params())

    def compute_feature_likelihood(self, X, probability_params):
        # likelihood = np.array([gmm_pdf(X, probability_params[0][prob_params_index], probability_params[1][prob_params_index], probability_params[2][prob_params_index]) for prob_params_index in range(self.k)])
        likelihood = gmm_pdf(X, *probability_params)
        likelihood = likelihood.sum(axis=1)
        return likelihood

    def get_likelihoods(self, X, label):
        feature_likelihood = np.array([self.compute_feature_likelihood(X[:, i], self.probability_parameters[label][i]) for i in range(X.shape[1])]).T
        likelihood = feature_likelihood.prod(axis=1)
        # likelihood = np.apply_along_axis(lambda i: np.apply_along_axis(lambda j: gmm_pdf(X, **self.probability_parameters[label][i]), 0, np.array(range(self.k))), 0, np.array(range(X.shape[1])))
        # likelihood = likelihood.sum(axis=0)
        # likelihood = likelihood.prod()
        return likelihood
        # for feature in range(X.shape[1]):
        #     feature_liklihood = 0
        #     for i in range(self.k):
        #         feature_liklihood += gmm_pdf(X, **self.probability_parameters[label][feature])

    def get_posterior(self, X, label):
        label_index = self.classes.index(label)
        return self.get_likelihoods(X, label) * self.prior[label_index]

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        posteriors = np.array([self.get_posterior(X, label) for label in self.classes])
        # posteriors = np.apply_along_axis(lambda label: self.get_posterior(X, label), 0,  np.array(self.classes)[:, np.newaxis])
        preds = np.apply_along_axis(lambda x: self.classes[np.argmax(x)], 1, posteriors.T)
        return preds


def calculate_accuracy(X, y, model):
    return y[model.predict(X) == y].shape[0] / X.shape[0]


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    lor_hypothesis = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_hypothesis.fit(x_train, y_train)
    lor_train_acc = calculate_accuracy(x_train, y_train, lor_hypothesis)
    lor_test_acc = calculate_accuracy(x_test, y_test, lor_hypothesis)

    naiveBayesGaussian_hypothesis = NaiveBayesGaussian(k)
    naiveBayesGaussian_hypothesis.fit(x_train, y_train)
    bayes_train_acc = calculate_accuracy(x_train, y_train, naiveBayesGaussian_hypothesis)
    bayes_test_acc = calculate_accuracy(x_test, y_test, naiveBayesGaussian_hypothesis)

    print(f'lor train acc: {lor_train_acc}')
    print(f'lor test acc: {lor_test_acc}')
    print(f'bayes train acc: {bayes_train_acc}')
    print(f'bayes test acc: {bayes_test_acc}')

    plt.title('lor decision regions')
    plot_decision_regions(x_train, y_train, lor_hypothesis)
    plt.title('bayes decision regions')
    plot_decision_regions(x_train, y_train, naiveBayesGaussian_hypothesis)

    plt.title('lor cost vs iteration')
    plt.plot(np.arange(len(lor_hypothesis.Js)), lor_hypothesis.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss as a function of iterations')
    plt.show()

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    dataset_a_features = np.vstack((multivariate_normal.rvs([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 250),
                                    multivariate_normal.rvs([5, 5, 5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 250),
                                    multivariate_normal.rvs([0, 0, 5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 250),
                                    multivariate_normal.rvs([5, 5, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 250)))
    dataset_a_labels = np.hstack((np.ones(500), np.zeros(500)))

    dataset_b_features = np.vstack((multivariate_normal.rvs([0, 0, 0], [[1, 0.3, 0.8], [0.3, 1, 0.8], [0.3, 0.3, 1]], 500),
                                    multivariate_normal.rvs([5, 5, 5], [[1, 0.3, 0.8], [0.3, 1, 0.8], [0.3, 0.3, 1]], 500),))
    dataset_b_labels = np.hstack((np.ones(500), np.zeros(500)))

    figure = plt.figure()
    figure.suptitle('a dataset that work better with naive bayes', fontsize=16)
    ax = plt.axes(projection="3d")
    ax.set_title(
        'the reason this dataset is better with naiveBayes is because linear regression is a linear seperator algorithm\nand our data set here is not linearly seperable.',
        fontsize=14)
    ax.scatter3D(dataset_a_features[:, 0], dataset_a_features[:, 1], dataset_a_features[:, 2],
                 c=np.vectorize({0: 'blue', 1: 'red'}.get)(dataset_a_labels))
    ax.set_xlabel('x feature', fontsize=14, color='purple')
    ax.set_ylabel('y feature', fontsize=14, color='red')
    ax.set_zlabel('z feature', fontsize=14, color='green')

    figure2 = plt.figure()
    figure2.suptitle('a dataset that work better with logistic regression', fontsize=16)
    ax2 = plt.axes(projection="3d")
    ax2.set_title(
        'the reason this dataset works better with logistic regression is first of all the fact that the data is linearly seperable.\nthe second reason is the fact naive bayes assumes that the rvs are independent, and clearly we can see that its not true.',
        fontsize=14)
    ax2.scatter3D(dataset_b_features[:, 0], dataset_b_features[:, 1], dataset_b_features[:, 2],
                  c=np.vectorize({0: 'blue', 1: 'red'}.get)(dataset_b_labels))
    ax2.set_xlabel('x feature', fontsize=14, color='purple')
    ax2.set_ylabel('y feature', fontsize=14, color='red')
    ax2.set_zlabel('z feature', fontsize=14, color='green')

    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }