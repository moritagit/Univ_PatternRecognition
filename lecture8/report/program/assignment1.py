# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cvxopt


def load_data(path):
    data = loadmat(path)
    #print(data.keys())
    x = data['x'].T
    l = data['l'].T
    n = data['n'][0, 0]
    d = data['d'][0, 0]
    return x, l, n, d


def inner_prod(x1, x2):
    value = x1.T.dot(x2)
    return value


def gauss_kernel(x1, x2, h):
    value = (1/2) * np.exp(-(x1 - x2)**2 / (2*h**2))
    return value


def solve_svm(x, y, kernel=inner_prod, eps=1e-5):
    n = y.shape[0]
    #K = kernel(x.T, x.T)
    #P = K * y.dot(y.T)
    h = x * y
    P = h.dot(h.T)
    qpP = cvxopt.matrix(P)
    qpq = cvxopt.matrix(-np.ones(n), (n, 1))
    qpG = cvxopt.matrix(-np.eye(n))
    qph = cvxopt.matrix(np.zeros(n), (n, 1))
    qpA = cvxopt.matrix(y.T.astype(float), (1, n))
    qpb = cvxopt.matrix(0.)

    cvxopt.solvers.options['abstol'] = eps

    res = cvxopt.solvers.qp(qpP, qpq, qpG, qph, qpA, qpb)
    alpha = np.reshape(np.array(res['x']), -1)[:, np.newaxis]

    w = np.sum(x * ((y*alpha) * np.ones(n)[:, np.newaxis]), axis=1)
    sv = (alpha > eps)
    isv = np.where(sv)[-1]
    b = np.sum(w.T.dot(x[:, isv]) -y[isv]) / np.sum(sv)

    return w, b, alpha


def plot(x, y, w, b, alpha, eps=1e-5, fig_path=None):
    plt.figure()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    sv = (alpha > eps)
    plt.plot(x[np.where((y>0) & sv), 0], x[np.where((y>0) & sv), 1], 'bo')
    plt.plot(x[np.where((y>0) & ~sv), 0], x[np.where((y>0) & ~sv), 1], 'bx')
    plt.plot(x[np.where((y<0) & sv), 0], x[np.where((y<0) & sv), 1], 'ro')
    plt.plot(x[np.where((y<0) & ~sv), 0], x[np.where((y<0) & ~sv), 1], 'rx')

    if abs(w[0]) > abs(w[1]):
        plt.plot([-1, 1],[(b+1+w[0])/w[1], (b+1-w[0])/w[1]])
        plt.plot([-1, 1],[(b-1+w[0])/w[1], (b-1-w[0])/w[1]])
    else:
        plt.plot([(b+1+w[1])/w[0], (b+1-w[1])/w[0]], [-1, 1])
        plt.plot([(b-1+w[1])/w[0], (b-1-w[1])/w[0]], [-1, 1])

    if fig_path:
        plt.savefig(fig_path)
    plt.show()


def main():
    # settings
    eps = 1e-5
    data_type = 'slinear'
    #data_type = 'qlinear'
    #data_type = 'slinear'
    #data_type = 'nonlinear'
    data_path = f'../data/{data_type}-data.mat'
    fig_path = f'../figures/assignment1_{data_type}_result.png'
    np.random.seed(0)
    cvxopt.solvers.options['reltol'] = 1e-10
    cvxopt.solvers.options['show_progress'] = False


    # load data
    x, y, n, d = load_data(data_path)
    print(f'Data Type: {data_type}  #Sample: {n}  #Dim: {d}\n')


    # calc
    kernel = inner_prod
    w, b, alpha = solve_svm(x, y, kernel=kernel, eps=eps)


    # result
    print(f'w = {w}  b = {b}')
    #print('alpha =', alpha)

    plot(x, y, w, b, alpha, eps=eps, fig_path=fig_path)


if __name__ == '__main__':
    main()
