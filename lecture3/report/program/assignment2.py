# -*- coding: utf-8 -*-


import pathlib
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

import mnread


def load_data():
    x_train = mnread.readim(mnread.trdatafz)
    y_train = mnread.readlabel(mnread.trlabelfz)
    x_test = mnread.readim(mnread.tstdatafz)
    y_test = mnread.readlabel(mnread.tstlabelfz)
    return x_train, y_train, x_test, y_test


def get_statistics(data, labels):
    n_data = len(labels)
    all_labels = sorted(list(set(labels)))
    # n_label = len(all_labels)
    statistics = {}
    for label in all_labels:
        _data = data[np.where(labels == label), :]
        n = _data.shape[1]
        _data = np.reshape(_data, [n, -1])
        statistics[label] = {
            'n': n,
            'p': n/n_data,
            'mean': np.mean(_data, axis=0),
            'cov': np.cov(_data.T),
            }
    return statistics


def train(case, statistics):
    all_labels = list(statistics.keys())
    n_label = len(all_labels)
    if case == 1:
        sigma = 0.0
        for label in all_labels:
            cov = statistics[label]['cov']
            sigma += np.diag(cov).mean()
        sigma /= n_label
        for label in all_labels:
            statistics[label]['cov_train'] = np.sqrt(sigma)
        log_prob = log_prob_1
    elif case == 2:
        Sigma = np.zeros_like(statistics[all_labels[0]]['cov'])
        for label in all_labels:
            cov = statistics[label]['cov']
            Sigma += cov
        Sigma /= n_label
        for label in all_labels:
            statistics[label]['cov_train'] = Sigma
        log_prob = log_prob_2
    elif case == 3:
        for label in all_labels:
            statistics[label]['cov_train'] = statistics[label]['cov']
        log_prob = log_prob_3
    else:
        raise ValueError("'case' must be 1, 2, or 3.")
    return statistics, log_prob


def log_probability_normal_dist(x, mean, cov, prior_prob,):
    d = len(mean)
    cov_inv = np.linalg.pinv(cov)
    logp = (
        -(1/2) * ((x - mean).dot(cov_inv) * (x - mean)).sum(axis=1)
        -(1/2) * np.log(np.linalg.det(cov))
        -(d/2) * np.log(2 * np.pi)
        + np.log(prior_prob)
        )
    return logp


def log_prob_1(x, mean, sigma, prior_prob, eps=1e-4):
    logp = mean.T.dot(x.T) - (1/2)*mean.T.dot(mean) + (sigma**2)*np.log(prior_prob)
    return logp


def log_prob_2(x, mean, cov, prior_prob, eps=1e-4):
    #cov_new = cov + eps*np.eye(len(cov))
    #cov_inv = np.linalg.inv(cov_new)
    cov_inv = np.linalg.pinv(cov)
    logp = mean.T.dot(cov_inv).dot(x.T) - (1/2)*mean.T.dot(cov_inv).dot(mean) + np.log(prior_prob)
    return logp


def log_prob_3(x, mean, cov, prior_prob, eps=1e-4):
    #cov_new = cov + eps*np.eye(len(cov))
    #cov_inv = np.linalg.inv(cov_new)
    #det = np.linalg.det(cov_new)
    #print(det, np.linalg.det(cov))
    cov_inv = np.linalg.pinv(cov)
    logp = -(1/2) * ((x - mean).dot(cov_inv) * (x - mean)).sum(axis=1)
    #logp += - (1/2)*np.log(det)
    logp += np.log(prior_prob)
    return logp


def classify(x, statistics, log_prob, eps=1e-4):
    n_data = len(x)
    x = np.reshape(x, [n_data, -1])
    all_labels = list(statistics.keys())
    y_pred = []
    for label in all_labels:
        mean = statistics[label]['mean']
        cov = statistics[label]['cov_train']
        prior_prob = statistics[label]['p']
        logp = log_prob(x, mean, cov, prior_prob, eps)
        y_pred.append(logp)
    y_pred = np.array(y_pred).T
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred


def evaluate(y_true, y_pred, labels):
    n_data = len(y_true)
    n_label = len(labels)

    # accuracy
    n_correct = (y_pred == y_true).sum()
    acc = n_correct / n_data
    print('All\t#Data: {}\t#Correct: {}\tAcc: {:.3f}'.format(n_data, n_correct, acc))

    # acc per label
    confusion_matrix = np.zeros((n_label, n_label), dtype=int)
    for i, label_true in enumerate(labels):
        idx_y_true = (y_true == label_true)
        _y_pred = y_pred[idx_y_true]
        _n_data = len(_y_pred)
        for j, label_pred in enumerate(labels):
            n = (_y_pred == label_pred).sum()
            confusion_matrix[i, j] = n

        n_correct = (_y_pred == label_true).sum()
        acc = n_correct / _n_data
        print('Label: {}\t#Data: {}\t#Correct: {}\tAcc: {:.3f}'.format(label_true, _n_data, n_correct, acc))
    print()
    print('Confusion Matrix\n', confusion_matrix)
    print()
    return confusion_matrix


def visualize(case, x, y_true, y_pred, image_dir, n=50):
    image_dir = pathlib.Path(image_dir)

    plt.figure()
    plt.suptitle('correct')
    indices_correct = np.random.permutation(np.where(y_pred==y_true)[-1])[range(n)]
    for i, idx_correct in enumerate(indices_correct):
        plt.subplot(5, 10, i+1)
        plt.axis('off')
        plt.imshow(x[idx_correct, :, :], cmap='gray')
        plt.title(y_pred[idx_correct])
    plt.savefig(str(image_dir / 'result_assignment2_case{}_correct.png'.format(case)))

    plt.figure()
    plt.suptitle('wrong')
    indices_wrong = np.random.permutation(np.where(~(y_pred==y_true))[-1])[range(n)]
    for i, idx_wrong in enumerate(indices_wrong):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(x[idx_wrong, :, :], cmap='gray')
        plt.title('{} ({})'.format(y_pred[idx_wrong], y_true[idx_wrong]))
    plt.savefig(str(image_dir / 'result_assignment2_case{}_wrong.png'.format(case)))
    plt.show()


def main():
    # settings
    case = 3
    image_dir = pathlib.Path().cwd().parent / 'figures'
    np.random.seed(0)
    print('Case: {}'.format(case))

    # load data
    x_train, y_train, x_test, y_test = load_data()
    #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    all_labels = sorted(list(set(y_train)))
    #n_data = len(y_train)
    #n_label = len(all_labels)
    #print(n_data, n_label, all_labels)

    # train (get statistics and calc sigmas)
    train_statistics = get_statistics(x_train, y_train)
    train_statistics, log_prob = train(case, train_statistics)

    # test
    y_pred = classify(x_test, train_statistics, log_prob, eps=1e-2)
    evaluate(y_test, y_pred, all_labels)
    visualize(case, x_test, y_test, y_pred, image_dir, n=50)


if __name__ == '__main__':
    main()
