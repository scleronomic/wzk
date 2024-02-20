import numpy as np


def maximum_a_posteriori(delta_y, delta_theta, cov_y_inv, cov_theta_inv=None, sum=True):
    if cov_theta_inv is None:
        return dCd(delta_y, cov_y_inv, sum=sum)

    else:
        return dCd(delta_y, cov_y_inv, sum=sum) + dCd(delta_theta, cov_theta_inv, sum=sum)


def dCd(delta, C, sum=True):
    if np.ndim(delta) + 1 == C.ndim:
        cdc = delta[..., :, np.newaxis] * C @ delta[..., :, np.newaxis]

    elif np.ndim(delta) == C.ndim == 2:
        cdc = delta[:, :, np.newaxis] * C[np.newaxis, :, :] @ delta[:, :, np.newaxis]

    else:
        raise ValueError(f"unknown sizes {delta.shape} & {C.shape}")

    if sum:
        return np.sum(cdc)
    else:
        return cdc
