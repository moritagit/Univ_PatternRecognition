# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load(path):
    data = loadmat(path)
    x = data['x']
    cov1 = data['cov1']
    cov2 = data['cov2']
    m1 = data['m1']
    m2 = data['m2']
    return x, m1, m2, cov1, cov2


def print_data(x, m1, m2, cov1, cov2):
    print('data')
    print('x shape: ', x.shape)
    print()
    print('data 1')
    print('m1 {}: \n'.format(m1.shape), m1)
    print('cov1 {}: \n'.format(cov1.shape), cov1)
    print()
    print('data 2')
    print('m2 {}: \n'.format(m2.shape), m2)
    print('cov2 {}: \n'.format(cov2.shape), cov2)
    print()


def sampling_norm_dist(x, y, mean, cov):
    icov = np.linalg.inv(cov)
    xt = x - mean[0, 0]
    yt = y - mean[0, 1]
    p = (
        1./(2. * np.pi * np.sqrt(np.linalg.det(cov)))
        * np.exp(-(1./2.) * (icov[0,0]*xt*xt + (icov[0,1] + icov[1,0])*xt*yt + icov[1,1]*yt*yt))
        )
    return p


def plot_data(
        x, m1, m2, cov1, cov2,
        x_linespace=None, y_linespace=None,
        show=True, save=False, path=None,
        ):
    if x_linespace is None:
        x_linespace = np.linspace(-10, 10, 100)
    if y_linespace is None:
        y_linespace = np.linspace(-10, 10, 100)

    x1, x2 = np.meshgrid(x_linespace, y_linespace)
    p1 = sampling_norm_dist(x1, x2, m1, cov1)
    p2 = sampling_norm_dist(x1, x2, m2, cov2)

    plt.figure(figsize=(10, 10))
    plt.axis('equal')
    plt.scatter(x[:, 0], x[:, 1], s=5)
    cs1 = plt.contour(x1, x2, p1, cmap='hsv')
    plt.clabel(cs1)
    cs2 = plt.contour(x1, x2, p2, cmap='hsv')
    plt.clabel(cs2)

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    return


def simultaneos_diagonalize(x, m1, m2, cov1, cov2):
    theta, phi = np.linalg.eig(cov1)
    # theta = np.diag(theta)
    theta_inv_half = np.diag(theta**(-1/2))
    k = (theta_inv_half).dot(phi.T).dot(cov2).dot(phi).dot(theta_inv_half)
    lamb, psi = np.linalg.eig(k)
    lamb = np.diag(lamb)

    # convert
    def convert(x):
        y = theta_inv_half.dot(phi.T).dot(x.T).T
        z = psi.T.dot(y.T).T
        return z

    z = convert(x)
    m1_z = convert(m1)
    m2_z = convert(m2)
    cov1_z = np.eye(x.shape[1])
    cov2_z = lamb
    return z, m1_z, m2_z, cov1_z, cov2_z


def main():
    # settings
    data_path = '../data/data2.mat'
    fig_path_before = '../figures/scatter_before_2.png'
    fig_path_after = '../figures/scatter_after_2.png'
    offset = 0.5

    print('data file: {}'.format(data_path))

    # load and look data
    x, m1, m2, cov1, cov2 = load(data_path)
    print_data(x, m1, m2, cov1, cov2)
    x_linespace = np.linspace(
        x[:, 0].min()-offset, x[:, 0].max()+offset, 100
        )
    y_linespace = np.linspace(
        x[:, 1].min()-offset, x[:, 1].max()+offset, 100
        )
    plot_data(
        x, m1, m2, cov1, cov2,
        x_linespace, y_linespace,
        save=True, path=fig_path_before,
        )

    # simultaneos diagonalization and convert
    z, m1_z, m2_z, cov1_z, cov2_z = simultaneos_diagonalize(x, m1, m2, cov1, cov2)
    print_data(z, m1_z, m2_z, cov1_z, cov2_z)
    x_linespace = np.linspace(
        z[:, 0].min()-offset, z[:, 0].max()+offset, 100
        )
    y_linespace = np.linspace(
        z[:, 1].min()-offset, z[:, 1].max()+offset, 100
        )
    plot_data(
        z, m1_z, m2_z, cov1_z, cov2_z,
        x_linespace, y_linespace,
        save=True, path=fig_path_after,
        )


if __name__ == '__main__':
    main()
