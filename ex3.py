import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # Extract prior distribution variables
    mu_prior = model.mu
    cov_prior = model.cov
    noise_var = model.sig

    # Fit the model to get posterior distribution variables
    model.fit(X, y)
    mu_posterior = model.fit_mu
    cov_posterior = model.fit_cov
    H = model.h(X)
    # calculate the log-evidence

    # log determinants:
    sign_prior, logdet_prior = np.linalg.slogdet(cov_prior)
    sign_post, logdet_post = np.linalg.slogdet(cov_posterior)

    # Evidence terms
    log_det_term = 0.5 * (logdet_post - logdet_prior)
    quadratic_term = -0.5 * ((mu_posterior - mu_prior).T @ np.linalg.inv(cov_prior) @ (mu_posterior - mu_prior))
    residuals = y - H @ mu_posterior
    data_fit_term = -0.5 * ((1 / noise_var) * np.sum(residuals ** 2) + X.shape[0] * np.log(noise_var))
    normalization_term = -(H.shape[0] / 2) * np.log(2 * np.pi)

    return log_det_term + quadratic_term + data_fit_term + normalization_term


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x ** 2 - 1
    f2 = lambda x: (-x ** 2 + 10 * x ** 3 + 50 * np.sin(x / 6) + 10) / 100
    f3 = lambda x: (.5 * x ** 6 - .75 * x ** 4 + 2.75 * x ** 2) / 50
    f4 = lambda x: 5 / (1 + np.exp(-4 * x)) - (x - 2 > 0) * x
    f5 = lambda x: 1 * (np.cos(x * 4) + 4 * np.abs(x - 2))
    functions = [f1, f2, f3, f4, f5]

    noise_var = .25
    x = np.linspace(-3, 3, 500)

    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alpha = 1

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))

        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            # <your code here>

        # plot evidence versus degree and predicted fit
        # <your code here>

    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162024.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        # <your code here>

    # plot log-evidence versus amount of sample noise
    # <your code here>


if __name__ == '__main__':
    main()



