# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load(path):
    data = loadmat(path)
    # print(data.keys())
    x1 = data['x1']
    x2 = data['x2']
    return x1, x2


def plot_data(x1, x2):
    # x1
    plt.hist(x1, bins=50)
    plt.savefig('../figures/x1.png')
    plt.show()

    # x2
    plt.hist(x2, bins=50)
    plt.savefig('../figures/x2.png')
    plt.show()


def normal_distribution(x, mu, sigma):
    p = (1 / np.sqrt(2*np.pi*sigma**2)) * np.exp(-(1/2) * ((x-mu)/sigma)**2)
    return p


def hypercube(x, mu, h):
    p = 1/h * (np.abs((x - mu)/h) < 1/2)
    return p


def conditional_probability_parzen(x, h, x_axis, kernel):
    n = len(x)
    hn = h/np.sqrt(n)
    prob = np.zeros(len(x_axis))
    for x_i in x:
        prob += kernel(x_axis, x_i, hn)
    prob = prob / n
    return prob


def conditional_probability_kmeans(x, k, x_axis, kernel):
    n = len(x)
    # k = np.sqrt(n)
    # kn = k/np.sqrt(n)
    prob = np.zeros(len(x_axis))

    for i in range(len(x_axis)):
        # r: sorted list by the distance to x[j]
        r = sorted(abs(x - x_axis[i]))

        # r[int(k)-1]: k-th distance
        prob[i] = k / (n * 2 * r[int(k)-1])

    return prob


def _nonparametric_method(
        x1, x2, p1, p2,
        param_candidates, param_str,
        kernel,
        conditional_probability,
        offset=1.0, num=100,
        path=None):
    x_min = min(x1.min(), x2.min()) - offset
    x_max = max(x1.max(), x2.max()) + offset
    x_axis = np.linspace(x_min, x_max, num)

    n_params = len(param_candidates)
    n_row = 3
    n_col = n_params + 1
    fig = plt.figure(figsize=(n_col*4, n_row*6))
    fig_idx = 0
    for i, text in enumerate(['prior probability (x1)', 'prior probability (x2)', 'posterior probability']):
        fig_idx = 1 + i*n_col
        ax = fig.add_subplot(n_row, n_col, fig_idx)
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=False,
            left=False,
            right=False,
            top=False,
            )
        for pos in ['bottom', 'left', 'right', 'top']:
            ax.spines[pos].set_visible(False)
        ax.text(0.5, 0.5, text, ha='center', va='bottom', fontsize=12)

    fig_idx = 0
    for i, param in enumerate(param_candidates):
        # calc conditional probability
        p1_cond = conditional_probability(
            x1, param, x_axis, kernel
            )
        p2_cond = conditional_probability(
            x2, param, x_axis, kernel
            )

        # calc post prob
        p1_joint = p1_cond * p1
        p2_joint = p2_cond * p2
        p_sum = p1_joint.sum() + p2_joint.sum()
        p1_post = p1_joint / p_sum
        p2_post = p2_joint / p_sum

        # plot
        ax_1 = fig.add_subplot(n_row, n_col, (i+1)+1)
        title = ''
        if param_str:
            title = '${} = {}$'.format(param_str, param)
        else:
            title = '{}'.format(param)
        ax_1.set_title(title)
        ax_1.hist(x1, bins=50, normed=True)
        ax_1.plot(x_axis, p1_cond)
        ax_1.set_ylim([0, 1.0])

        ax_2 = fig.add_subplot(n_row, n_col, n_col+(i+1)+1)
        ax_2.hist(x2, bins=50, normed=True)
        ax_2.plot(x_axis, p2_cond)
        ax_2.set_ylim([0, 1.0])

        ax = fig.add_subplot(n_row, n_col, 2*n_col+(i+1)+1)
        ax.plot(x_axis, p1_post, label='1')
        ax.plot(x_axis, p2_post, label='2')
        ax.legend()
        # ax.set_ylim([0, 1.0])

    plt.savefig(str(path))
    plt.show()


def parzen(x1, x2, p1, p2, h_list, kernel, offset=1.0, num=100, path=None):
    _nonparametric_method(
        x1, x2, p1, p2,
        h_list, 'h',
        kernel,
        conditional_probability_parzen,
        offset, num, path
        )


def kmeans(x1, x2, p1, p2, k_list, kernel, offset=1.0, num=100, path=None):
    _nonparametric_method(
        x1, x2, p1, p2,
        k_list, 'k',
        kernel,
        conditional_probability_kmeans,
        offset, num, path
        )


def main():
    # settings
    data_path = '../data/data.mat'
    offset = 1.0
    num = 100
    np.random.seed(0)


    # load data
    x1, x2 = load(data_path)
    #print(x1.shape, x2.shape)
    x1, x2 = x1[0], x2[0]

    #plot_data(x1, x2)
    n1, n2 = len(x1), len(x2)
    n = n1 + n2
    p1, p2 = n1/n, n2/n


    # parzen
    """
    # kernel = normal_distribution
    kernel = hypercube
    h_list = [1.0, 3.0, 5.0, 10.0]
    fig_path = '../figures/parzen_hypercube_result.png'

    parzen(
        x1, x2, p1, p2,
        h_list, kernel,
        offset, num,
        fig_path
        )
    """


    # kmeans
    k_list = [2, 5, 10, 14, 20]
    fig_path = '../figures/kmeans_result.png'

    kmeans(
        x1, x2, p1, p2,
        k_list, None,
        offset, num,
        fig_path
        )
    #"""


if __name__ == '__main__':
    main()
