"""
First-order convex optimization solvers that can be used for inferring sparse coefficients in dictionary learning.

@Filename    solvers
@Author      Kion 
@Created     5/29/20
"""
import numpy as np


def soft_thresh(x, tau):
    """
    Proximal gradient for L1 norm
    :param x: Vector to evaluate proximal gradient step for
    :param tau: L1 regularisation constant
    :return: Vector after L1 proximal gradient step
    """
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.)


def FISTA(A, y, tau=1e-2, max_iter=500, tol=1e-3):
    """
    Fast Iterative-Shrinkage Thresholding algorithm from (Beck et al. 2009)
    :param A: Observation matrix, in this case the dictionary
    :param y: Measurements, in this case the image patch
    :param tau: L1 regularisation constant
    :param max_iter: Max amount of iterations to run solver
    :param tol: Minimum amount of change in solution for exit condition
    :return: Solution to L1 optimization program
    """
    x = A.T @ y
    z = np.array(x)
    k = 0
    L = np.linalg.norm(A, ord=2) ** 2
    t = 1
    change = 1e99

    # Check exit condition
    while k < max_iter and change > tol:
        # Save copy of current iteration for calculating change
        old_x = np.array(x)

        # Take gradient step to minimize L2 term
        x = z - (1 / L) * A.T @ (A @ z - y)
        # Take proximal gradient step to minimize L1 term
        x = soft_thresh(x, (tau / L))

        # Compute new momentum weighting term
        new_t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        # Compute momentum as weighted sum of current and past iteration
        z = x + ((t - 1) / new_t) * (x - old_x)

        # Increment iteration, calculate change for exit criterion
        k += 1
        t = new_t
        change = np.linalg.norm(x - old_x) / np.linalg.norm(old_x)
    return x


def ADMM(A, y, tau=1e-2, rho=10, max_iter=500, eps_abs=1e-2, eps_rel=1e-3):
    """
    Alternating Direction Method of Multipliers. See review by Boyd.
    :param A: Observation matrix, in this case the dictionary
    :param y: Measurements, in this case the image patch
    :param tau: L1 regularisation constant
    :param rho: Regularisation term used in the augmented Lagrangian to trade-off primal and dual feasibility
    :param max_iter: Max amount of iterations to run solver for
    :param eps_abs: Absolute tolerance for exit criterion
    :param eps_rel: Relative tolerance for exit criterion. Change depending on signal magnitude
    :return: Solution to L1 optimization program
    """
    x = A.T @ y
    z = 0
    mu = 0
    k = 0

    primal_gap = 1e99
    dual_gap = 1e99
    eps_primal = np.sqrt(y.shape[0]) * eps_abs
    eps_dual = np.sqrt(x.shape[0]) * eps_abs

    # Cache the matrix inverse so we do not solve every iteration
    A_mtx_inv = np.linalg.inv(A.T @ A + rho * np.eye(A.shape[1]))

    while k < max_iter and (primal_gap > eps_primal or dual_gap > eps_dual):
        z_old = np.array(z)

        mu = mu + x - z
        x = A_mtx_inv @ (A.T @ y + rho * (z - mu))
        z = soft_thresh(x + mu, tau / rho)

        eps_primal = np.sqrt(y.shape[0]) * eps_abs + np.maximum(np.linalg.norm(x), np.linalg.norm(z)) * eps_rel
        eps_dual = np.sqrt(x.shape[0]) * eps_abs + np.linalg.norm(rho * mu) * eps_rel

        primal_gap = np.linalg.norm(x - z)
        dual_gap = np.linalg.norm(rho * (z - z_old))

        k += 1
    return x
