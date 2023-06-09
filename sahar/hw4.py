import sys

import numpy as np


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

    def apply_bias_trick(self, X):
        """
        Applies the bias trick to the input data.

        Input:
        - X: Input data (m instances over n features).

        Returns:
        - X: Input data with an additional column of ones in the
            zeroth position (m instances over n+1 features).
        """

        ones_array = np.ones((len(X), 1))
        # if array is 1D array case
        newX = X.reshape(-1, 1) if X.ndim == 1 else X
        X = np.concatenate((ones_array, newX), axis=1)

        return X

    def sigmoid(self, x):
        """
        Calculates the sigmoid function for a given data table and the current model parameters

        Parameters
        ----------
        x : an instance in the data

        Returns
        -------
        the calculated sigmoid function result

        """
        # x = self.theta.reshape(-1, 1).T.dot(x)
        # x = x.dot(self.theta.reshape(1, -1).T)
        x = x.dot(self.theta)
        return 1 / (1 + np.exp(-x))

    def cost_function(self, X, y):
        """
        Calculates the cost function value for a given data table and labels.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        the value of the cost function

        """
        instances = X.shape[0]
        # sigmoid = np.apply_along_axis(self.sigmoid, 1, X)
        sigmoid = self.sigmoid(X)
        epsilon = 1e-5
        cost = (-y * np.log(sigmoid + epsilon) - (1 - y) @ np.log(1 - sigmoid + epsilon))
        return np.sum(cost) / instances

    def cost_derivative(self, X, y):
        """
        Calculates the cost derivative value given an instance and label.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        the value of the cost function derivative

        """
        check = (self.sigmoid(X) - y).dot(X)
        return np.sum((self.sigmoid(X) - y).dot(X)) / X.shape[0]

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

        X = self.apply_bias_trick(X)

        try:
            features = X.shape[1]
        except:
            features = 1

        self.theta = np.random.rand(features)
        self.thetas.append(self.theta)
        # apply gradient descent
        for _ in range(self.n_iter):
            self.theta = self.theta - self.eta * self.cost_derivative(X, y)
            current_cost = self.cost_function(X, y)
            self.Js.append(current_cost)
            self.thetas.append(self.theta)
            if len(self.Js) > 1 and abs(current_cost - self.Js[-2]) < self.eps:
                break

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X = self.apply_bias_trick(X)
        sigmoid_values = self.sigmoid(X).reshape(-1, 1)
        preds = np.where(sigmoid_values < 0.5, 0, 1)
        return preds.reshape(-1, 1)

    def predict_instance(self, x):
        sigmoid = self.sigmoid(x).reshape(-1, 1)
        if sigmoid > 1 - sigmoid:
            return 1
        return 0


def cross_validation(X, y, folds, algo, random_state):
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

    # shuffle the array
    x_y = np.hstack((X, y.reshape(-1, 1)))
    np.random.seed(random_state)
    np.random.shuffle(x_y)

    acc_array = []
    split_arr = np.array_split(x_y, folds)
    test = split_arr[-1]

    for fold in split_arr[:-1]:
        algo.fit(fold[:, :-1], fold[:, -1])
        prediction = algo.predict(test[:, :-1])
        mask = (prediction == test[:, -1].reshape(-1, 1))
        sum_equal_rows = prediction[mask].sum()
        acc_array.append(sum_equal_rows / len(prediction))

    return np.mean(acc_array)


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    return p


class EM(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

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

        self.responsibilities = []
        self.weights = []
        self.mus = []
        self.sigmas = []
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        np.random.seed(self.random_state)
        num_samples, num_features = data.shape

        # pick k random indexes to be the initial mus
        index = np.random.choice(num_samples, self.k, replace=False)
        self.mus = data[index]
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
        self.responsibilities = np.zeros((self.k, int(data.shape[0])))
        for i in range(self.k):
            for j, row in enumerate(data):
                total_pdf = np.array(self.calculate_pdf_array(row))
                self.responsibilities[i, j] = (self.weights[i] * norm_pdf(row, self.mus[i], self.sigmas[i]).flatten()) / \
                                              sum(np.array(self.weights).dot(total_pdf))

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        num_samples = data.shape[0]
        # update mus
        for i in range(self.k):
            # update weights
            self.weights[i] = sum(self.responsibilities[i]) / num_samples
            # update mus
            self.mus[i] = sum(self.responsibilities[i].dot(data)) / (self.weights[i] * num_samples)
            # update sigmas
            self.sigmas[i] = np.sqrt(
                sum(np.array(self.responsibilities[i]).dot((data - self.mus[i]) ** 2)) /
                (self.weights[i] * num_samples))

    def log_likelihood(self, data):
        """
        Calculates the loglikelihood of the data given the current parameters.

        Parameters
        ----------
        data: the data we want to calculate log likelihood for.

        Returns
        -------
        loglikelihood of the data given the current parameters.
        """
        sum_all = 0
        for i in range(data.shape[0]):
            for j in range(self.k):
                internal_sum = 0
                internal_sum += self.weights[j] * norm_pdf(data[i], self.mus[j], self.sigmas[j])
            sum_all += np.log(internal_sum)
        return sum_all

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
        last_cost = 0
        new_cost = sys.maxsize
        for i in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            last_cost = new_cost
            new_cost = self.log_likelihood(data)
            if abs(new_cost - last_cost) < self.eps:
                break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


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
    pdf = 0
    for mixture in zip(weights, mus, sigmas):
        pdf += mixture[0] * norm_pdf(data, mixture[1], mixture[2])
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
        classes = np.unique(y)
        for one_class in classes:
            masked_data = X.loc[y == one_class]
            class_emm = EM(self.k)
            class_emm.fit(masked_data)
            




    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


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

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'dataset_a_features': dataset_a_features,
            'dataset_a_labels': dataset_a_labels,
            'dataset_b_features': dataset_b_features,
            'dataset_b_labels': dataset_b_labels
            }
