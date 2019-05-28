# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def dataset1():
    n = 1000
    alpha = 0.3

    # prior probability and number of samples
    n1 = sum(np.random.rand(n) < alpha)
    n2 = n - n1

    mean1, mean2 = np.array([2, 0]), np.array([-2, 0])
    cov1 = np.array([[1, 0], [0, 9]])
    cov2 = np.array([[1, 0], [0, 2]])

    # generate data
    x1 = sampling_normal_dist(mean1, cov1, n1)
    x2 = sampling_normal_dist(mean2, cov2, n2)

    return x1, x2, mean1, mean2, cov1, cov2


def dataset2():
    n = 1000
    alpha = 0.4

    # prior probability and number of samples
    n1 = sum(np.random.rand(n) < alpha)
    n2 = n - n1

    mean1, mean2 = np.array([2, 2]), np.array([-2, -2])
    cov1 = np.array([[5, 0], [0, 6]])
    cov2 = np.array([[6, 0], [0, 4]])

    # generate data
    x1 = sampling_normal_dist(mean1, cov1, n1)
    x2 = sampling_normal_dist(mean2, cov2, n2)

    return x1, x2, mean1, mean2, cov1, cov2


def dataset3():
    n = 1000
    alpha = 0.5

    # prior probability and number of samples
    n1 = sum(np.random.rand(n) < alpha)
    n2 = n - n1

    mean1, mean2 = np.array([1, 1]), np.array([1, 1])
    cov1 = np.array([[1, 0], [0, 2]])
    cov2 = np.array([[8, 0], [0, 5]])

    # generate data
    x1 = sampling_normal_dist(mean1, cov1, n1)
    x2 = sampling_normal_dist(mean2, cov2, n2)

    return x1, x2, mean1, mean2, cov1, cov2


def sampling_normal_dist(mean, cov, n):
    return np.random.multivariate_normal(mean, cov, n)


def log_probability_normal_dist(x, mean, cov, prior_prob,):
    d = len(mean)
    cov_inv = np.linalg.inv(cov)
    logp = (
        -(1/2) * ((x - mean).dot(cov_inv) * (x - mean)).sum(axis=1)
        -(1/2) * np.log(np.linalg.det(cov))
        -(d/2) * np.log(2 * np.pi)
        )
    return logp


def classifier_binary(x, a, b, c):
    result = (x.dot(a) * x).sum(axis=1)
    result += b.dot(x.T)
    result += c
    return result


def classify_binary_1(x, mean1, mean2, sigma, p1, p2):
    d = len(mean1)
    a = np.zeros((d, d))
    b = (1/sigma**2) * (mean1 - mean2)
    c = -(1/(2*(sigma)**2)) * (np.linalg.norm(mean1)**2 - np.linalg.norm(mean2)**2) + np.log(p1/p2)
    result = classifier_binary(x, a, b, c)
    return result, a, b, c


def classify_binary_2(x, mean1, mean2, cov, p1, p2):
    d = len(mean1)
    cov_inv = np.linalg.inv(cov)
    a = np.zeros((d, d))
    b = cov_inv.dot(mean1 - mean2)
    c = -(1/2) * (mean1.T.dot(cov_inv).dot(mean1) - mean2.T.dot(cov_inv).dot(mean2)) + np.log(p1/p2)
    result = classifier_binary(x, a, b, c)
    return result, a, b, c


def classify_binary_3(x, mean1, mean2, cov1, cov2, p1, p2):
    cov1_inv = np.linalg.inv(cov1)
    cov2_inv = np.linalg.inv(cov2)
    a = -(1/2) * (cov1_inv - cov2_inv)
    b = cov1_inv.dot(mean1)  - cov2_inv.dot(mean2)
    c = (
        -(1/2)*(mean1.T.dot(cov1_inv).dot(mean1) - mean2.T.dot(cov2_inv).dot(mean2))
        - (1/2)*np.log(np.linalg.det(cov1)/np.linalg.det(cov2))
        + np.log(p1/p2)
        )
    result = classifier_binary(x, a, b, c)
    return result, a, b, c


def measure_accuracy(result, label, n_data):
    if label == 1:
        is_correct = (result >= 0)
    elif label == 2:
        is_correct = (result < 0)
    else:
        raise ValueError("'label' must be 1 or 2.")

    n_correct = is_correct.sum()
    acc = n_correct / n_data
    print('#Data: {}\t#Correct: {}\tAcc: {:.3f}'.format(n_data, n_correct, acc))
    return is_correct


def sampling_normal_dist_for_contour(x, y, mean, cov):
    icov = np.linalg.inv(cov)
    xt = x - mean[0]
    yt = y - mean[1]
    p = (
        1./(2. * np.pi * np.sqrt(np.linalg.det(cov)))
        * np.exp(
            -(1./2.)*(
                icov[0, 0]*xt*xt
                + (icov[0, 1] + icov[1, 0])*xt*yt
                + icov[1, 1]*yt*yt
                )
            )
        )
    return p


def plot_data(
        x1_o, x1_x, x2_o, x2_x,
        a, b, c,
        mean1, mean2, cov1, cov2,
        x_linespace=None, y_linespace=None,
        show=True, save=False, path=None,
        size=10,
        ):
    if x_linespace is None:
        x_linespace = np.linspace(-10, 10, 100)
    if y_linespace is None:
        y_linespace = np.linspace(-10, 10, 100)
    _x, _y = np.meshgrid(x_linespace, y_linespace)

    plt.figure(figsize=(15, 15))
    plt.axis('equal')

    p1_contour = sampling_normal_dist_for_contour(_x, _y, mean1, cov1)
    plt.scatter(x1_o[:, 0], x1_o[:, 1], s=size, marker='o', color='darkorange')
    plt.scatter(x1_x[:, 0], x1_x[:, 1], s=size, marker='o', color='blue')
    cs1 = plt.contour(_x, _y, p1_contour, cmap='hsv')
    plt.clabel(cs1)

    p2_contour = sampling_normal_dist_for_contour(_x, _y, mean2, cov2)
    plt.scatter(x2_o[:, 0], x2_o[:, 1], s=size, marker='x', color='royalblue')
    plt.scatter(x2_x[:, 0], x2_x[:, 1], s=size, marker='x', color='red')
    cs2 = plt.contour(_x, _y, p2_contour, cmap='hsv')
    plt.clabel(cs2)

    # decision boundary
    _xy = np.c_[np.reshape(_x, -1), np.reshape(_y, -1)]
    pp = classifier_binary(_xy, a, b, c)
    pp = np.reshape(pp, _x.shape)
    cs = plt.contour(_x, _y, pp, levels=[0.0], colors=['darkcyan'])
    # plt.clabel(cs)

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    return


def main():
    # settings
    case = 3
    dataset_id = 3
    fig_path = '../figures/result_assignment1_dataset{}_case{}.png'.format(dataset_id, case)
    offset = 0.5
    n_linespace = 100
    np.random.seed(0)

    print('Case {}'.format(case))
    print('Dataset: {}'.format(dataset_id))
    print()


    # load data
    x1, x2, mean1, mean2, cov1, cov2 = dataset3()
    n1 = len(x1)
    n2 = len(x2)
    n = n1 + n2
    p1 = n1 / n
    p2 = n2 / n


    # decide which model to use
    if case == 1:
        sigma = (np.diag(cov1).sum() + np.diag(cov2).sum())/4
        result_1, a, b, c = classify_binary_1(
            x1, mean1, mean2, sigma, p1, p2
            )
        result_2, a, b, c = classify_binary_1(
            x2, mean1, mean2, sigma, p1, p2
            )
    elif case == 2:
        cov = (cov1 + cov2)/2
        result_1, a, b, c = classify_binary_2(
            x1, mean1, mean2, cov, p1, p2
            )
        result_2, a, b, c = classify_binary_2(
            x2, mean1, mean2, cov, p1, p2
            )
    elif case == 3:
        result_1, a, b, c = classify_binary_3(
            x1, mean1, mean2, cov1, cov2, p1, p2
            )
        result_2, a, b, c = classify_binary_3(
            x2, mean1, mean2, cov1, cov2, p1, p2
            )
    else:
        raise ValueError("'case' must be 1, 2, or 3.")


    # classify x1
    print('x1')
    is_correct_1 = measure_accuracy(result_1, 1, len(x1))
    x1_o = x1[is_correct_1]
    x1_x = x1[~is_correct_1]
    print()

    # classify x2
    print('x2')
    is_correct_2 = measure_accuracy(result_2, 2, len(x2))
    x2_o = x2[is_correct_2]
    x2_x = x2[~is_correct_2]
    print()

    acc = (is_correct_1.sum() + is_correct_2.sum()) / (len(is_correct_1) + len(is_correct_2))
    print('Accuracy: {:.3f}'.format(acc))
    print()


    # plot
    _x = np.concatenate([x1, x2], axis=0)
    x_linespace = np.linspace(
        _x[:, 0].min()-offset, _x[:, 0].max()+offset, n_linespace
        )
    y_linespace = np.linspace(
        _x[:, 1].min()-offset, _x[:, 1].max()+offset, n_linespace
        )
    plot_data(
        x1_o, x1_x, x2_o, x2_x,
        a, b, c,
        mean1, mean2, cov1, cov2,
        x_linespace, y_linespace,
        save=True, path=fig_path,
        size=20,
        )


if __name__ == '__main__':
    main()
