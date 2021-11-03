import logging

import numpy as np

def print_debug(train_args, b_true, b_hat):
    count_nz = np.zeros(train_args.dict_size + 1, dtype=int)
    coeff_nz = np.count_nonzero(b_hat.T, axis=0)
    total_nz = np.count_nonzero(b_hat.T, axis=1)
    true_total_nz = np.count_nonzero(b_true.T, axis=1)

    for z in range(len(total_nz)):
        count_nz[total_nz[z]] += 1
    mean_coeff = b_hat[np.nonzero(b_hat)].mean()
    true_coeff = b_true[np.nonzero(b_hat)].mean()

    coeff_acc = 0
    for k in range(len(b_hat)):
        true_sup = np.where(np.abs(b_true[k]) > 1e-6)[0]
        est_sup = np.where(np.abs(b_hat[k]) > 1e-6)[0]
        missed_support = np.setdiff1d(true_sup, est_sup)
        excess_support = np.setdiff1d(est_sup, true_sup)
        coeff_acc += (len(missed_support) + len(excess_support)) / b_true.shape[1]
    coeff_acc /= len(b_hat)

    logging.info(f"Mean est coeff support: {total_nz.mean():.3f}")
    logging.info(f"Mean true coeff support: {true_total_nz.mean():.3f}")
    logging.info(f"Mean est coeff magnitude: {mean_coeff}")
    logging.info(f"Mean true coeff magnitude: {true_coeff}")
    logging.info("L1 distance with true coeff: {:.3E}".format(np.abs(b_hat - b_true).sum()))
    logging.info("Coeff support accuracy: {:.2f}%".format(100.*coeff_acc))