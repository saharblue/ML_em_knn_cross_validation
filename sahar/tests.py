import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from hw4 import LogisticRegressionGD, cross_validation, EM
from sklearn.mixture import GaussianMixture


training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')
X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values

# #### Your code here ####
# etas = [0.05, 0.005, 0.0005, 0.00005, 0.000005]
# epss = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
# folds = 5
#
# random_state = 1
# acc_dict = {}
# for eta in etas:
#     for eps in epss:
#         lor = LogisticRegressionGD(eta=eta, eps=eps, random_state=random_state)
#         acc = cross_validation(X_training, y_training, folds, lor, random_state=random_state)
#         acc_dict[(eta, eps)] = acc
#         print(f"Accuracy with eta={eta}, eps={eps}:  {acc:.6f}")
#
# best_params = max(acc_dict, key=acc_dict.get)
# best_eta = best_params[0]
# best_eps = best_params[1]

em_obj = EM(k=2)
em_obj.fit(X_training[:, 0].reshape(-1, 1))

print(f"mus = {em_obj.get_dist_params()[1]}")
print(f"sigmas = {em_obj.get_dist_params()[2]}")
print(f"weights = {em_obj.get_dist_params()[0]}")


data = X_training[:, 0].reshape(-1, 1)
gmm = GaussianMixture(n_components=2, max_iter=1000, random_state=1991)
gmm.fit(data)

print(f"mus = {gmm.means_}")
print(f"sigmas = {gmm.covariances_}")
print(f"weights = {gmm.weights_}")
