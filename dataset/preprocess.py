import numpy as np
from copy import deepcopy


def add_lighting_noise(rgb_0to1, alpha_std):
    """
    Add AlexNet-style PCA-based noise

    :param rgb_0to1: Image in RGB format, normalized within [0, 1]; values can fall outside [0, 1] due to
                     some preceding processing, but eigenvalues/vectors should match the magnitude order
                     numpy array of shape (h, w, 3)
    :param alpha_std: Standard deviation of the Gaussian from which alpha is drawn positive float
    :return: rgb_0to1_out: Output image in RGB format, with lighting noise added numpy array of the same shape as input
    """
    eigvals = np.array((0.2175, 0.0188, 0.0045))
    eigvecs = np.array(((-0.5675, 0.7192, 0.4009), (-0.5808, -0.0045, -0.8140), (-0.5836, -0.6948, 0.4203)))

    alpha = np.random.normal(loc=0, scale=alpha_std, size=3)
    noise_rgb = np.sum(np.multiply(np.multiply(eigvecs, np.tile(alpha, (3, 1))), np.tile(eigvals, (3, 1))), axis=1)

    rgb_0to1_out = deepcopy(rgb_0to1)
    for i in range(3):
        rgb_0to1_out[:, :, i] += noise_rgb[i]

    return rgb_0to1_out


