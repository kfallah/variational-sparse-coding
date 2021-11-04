import logging

import numpy as np

def print_debug(train_args, b_true, b_hat):
    total_nz = np.count_nonzero(b_hat, axis=1)
    total_near_nz = np.abs(b_hat) > 1e-6
    true_total_nz = np.count_nonzero(b_true, axis=1)
    mean_coeff = np.abs(b_hat[np.nonzero(b_hat)]).mean()
    true_coeff = np.abs(b_true[np.nonzero(b_hat)]).mean()

    coeff_acc = 0
    for k in range(len(b_hat)):
        true_sup = np.nonzero(b_true[k])[0]
        est_sup = np.nonzero(b_hat[k])[0]
        missed_support = np.setdiff1d(true_sup, est_sup)
        excess_support = np.setdiff1d(est_sup, true_sup)
        coeff_acc += (b_true.shape[1] - len(missed_support) - len(excess_support)) / b_true.shape[1]
    coeff_acc /= len(b_hat)

    logging.info(f"Mean est coeff near-zero: {total_near_nz.sum(axis=-1).mean():.3f}")
    logging.info(f"Mean est coeff support: {total_nz.mean():.3f}")
    logging.info(f"Mean true coeff support: {true_total_nz.mean():.3f}")
    logging.info(f"Mean est coeff magnitude: {mean_coeff}")
    logging.info(f"Mean true coeff magnitude: {true_coeff}")
    logging.info("L1 distance with true coeff: {:.3E}".format(np.abs(b_hat - b_true).sum()))
    logging.info("Coeff support accuracy: {:.2f}%".format(100.*coeff_acc))