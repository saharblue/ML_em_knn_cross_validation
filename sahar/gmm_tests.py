from hw4 import gmm_pdf
import numpy as np
import scipy.stats as stats

def test_gmm_pdf():
    # Test 1: Single component, weight of 1
    data = np.array([1.5, 2.0, 3.5])
    weights = np.array([1.0])
    mus = np.array([2.0])
    sigmas = np.array([1.0])
    result = gmm_pdf(data, weights, mus, sigmas)
    expected = stats.norm.pdf(data, mus[0], sigmas[0])
    assert np.allclose(result, expected), f'Expected {expected}, but got {result}'

    # Test 2: Two components with equal weight
    weights = np.array([0.5, 0.5])
    mus = np.array([1.0, 3.0])
    sigmas = np.array([1.0, 0.5])
    result = gmm_pdf(data, weights, mus, sigmas)
    expected = 0.5 * stats.norm.pdf(data, mus[0], sigmas[0]) + 0.5 * stats.norm.pdf(data, mus[1], sigmas[1])
    assert np.allclose(result, expected), f'Expected {expected}, but got {result}'

    # Test 3: Two components with different weights
    weights = np.array([0.3, 0.7])
    result = gmm_pdf(data, weights, mus, sigmas)
    expected = 0.3 * stats.norm.pdf(data, mus[0], sigmas[0]) + 0.7 * stats.norm.pdf(data, mus[1], sigmas[1])
    assert np.allclose(result, expected), f'Expected {expected}, but got {result}'

test_gmm_pdf()



