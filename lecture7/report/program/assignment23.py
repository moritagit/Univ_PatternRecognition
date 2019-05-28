# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import mnread
from neural_network import *


def load_data():
    train_X = mnread.readim(mnread.trdatafz)
    train_X = np.reshape(train_X, [train_X.shape[0], -1])  #flatten
    train_y = mnread.readlabel(mnread.trlabelfz)

    test_X = mnread.readim(mnread.tstdatafz)
    test_X = np.reshape(test_X, [test_X.shape[0], -1])  #flatten
    test_y = mnread.readlabel(mnread.tstlabelfz)
    return train_X, train_y, test_X, test_y


def add_augment_axis(data_X):
    n = data_X.shape[0]
    data_X_augmented = np.concatenate((
        np.ones((n, 1)),
        data_X
        ), axis=1)
    return data_X_augmented


def normalize(data_X):
    data_X -= 128
    data_X /= 128
    return data_X


def split(data_X, data_y, batch_size):
    n_data = len(data_y)
    n_abandoned = n_data % batch_size
    if n_abandoned != 0:
        print(f'Warning: {n_abandoned} samples are abandoned')
    data_X_split = [data_X[i:i+batch_size] for i in range(0, n_data, batch_size)]
    data_y_split = [data_y[i:i+batch_size] for i in range(0, n_data, batch_size)]
    return data_X_split, data_y_split


def train(
        model, labels,
        train_X, train_y,
        valid_X, valid_y,
        lr, epochs, batch_size,
        mode='MSE',
        ):
    if mode.lower() not in ['mse', 'lms', 'widrow-hoff', 'mlp', 'nn']:
        raise ValueError(f'Unknown mode: {mode}')
    n_label = len(labels)
    train_X_split, train_y_split = split(train_X, train_y, batch_size)
    n_minibatch = len(train_y_split)
    print('train')
    for epoch in range(epochs):
        # train
        for i in range(n_minibatch):
            # forward
            y_pred = model(train_X_split[i], is_training=True)
            y_true = np.identity(n_label)[train_y_split[i]]
            loss = compute_loss(y_pred, y_true)

            # update
            delta = y_pred - y_true
            if mode.lower() in ['mse', 'lms', 'widrow-hoff']:
                model.compute_grad(delta)
            else:
                model.back_propagate(delta)
            model.update(lr)

        # validate
        y_pred = model(valid_X, is_training=False)
        loss = compute_loss(y_pred, np.identity(n_label)[valid_y])
        y_pred = np.argmax(y_pred, axis=1)
        n_correct = (y_pred == valid_y).sum()
        acc = n_correct / len(valid_y)

        print(f'Epoch: {epoch+1}    #Loss: {loss:.3f}    #Correct: {n_correct}    Accuracy: {acc:.3f}')
    print()
    return model


def test(model, test_X, test_y, labels):
    # settings
    n_label = len(labels)
    confusion_matrix = np.zeros((n_label, n_label), dtype=int)
    n_data_all = len(test_y)
    result = {}

    print('test')

    # prediction
    y_pred = model(test_X)
    y_pred = np.argmax(y_pred, axis=1)

    # calc scores
    for label in labels:
        print(f'Label: {label}\t', end='')

        indices = np.where(test_y == label)[-1]
        n_data = len(indices)
        preds = y_pred[indices]

        # make confusion matrix
        for i in labels:
            n = (preds == i).sum()
            confusion_matrix[label, i] = n

        # calc accuracy
        n_correct = confusion_matrix[label, label]
        acc = n_correct / n_data
        print(f'#Data: {n_data}\t#Correct: {n_correct}\tAcc: {acc:.3f}')

        result[label] = {
            'data': n_data,
            'correct': n_correct,
            'accuracy': acc,
            }
    result['confusion_matrix'] = confusion_matrix

    # overall score
    n_crr_all = np.diag(confusion_matrix).sum()
    acc_all = n_crr_all / n_data_all
    result['all'] = {
        'data': n_data_all,
        'correct': n_crr_all,
        'accuracy': acc_all,
        }
    print(f'All\t#Data: {n_data_all}\t#Correct: {n_crr_all}\tAcc: {acc_all:.3f}')
    print()
    print('Confusion Matrix:\n', confusion_matrix)
    print()
    return result


def print_result_in_TeX_tabular_format(result):
    labels = list(range(10))

    print('Scores')
    for label in labels:
        print('{} & {} & {} & {:.3f} \\\\'.format(
            label,
            int(result[label]['data']),
            int(result[label]['correct']),
            result[label]['accuracy']
            ))
    print('All & {} & {} & {:.3f} \\\\'.format(
        int(result['all']['data']),
        int(result['all']['correct']),
        result['all']['accuracy']
        ))
    print()

    print('Confusion Matrix')
    for i in labels:
        print('{}    '.format(i), end='')
        for j in labels:
            print(' & {}'.format(int(result['confusion_matrix'][i, j])), end='')
        print(' \\\\')
    return


def compute_loss(y_pred, y_true):
    loss = np.mean((y_pred - y_true)**2)
    return loss


class Linear(object):
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(input_size, output_size).astype(float)
        self.dW = None
        self.x = None

    def __call__(self, x, is_training=True):
        self.x = x
        y = x.dot(self.W)
        return y

    def compute_grad(self, delta):
        batch_size = delta.shape[0]
        self.dW = self.x.T.dot(delta) / batch_size

    def update(self, lr):
        self.W -= lr * self.dW


def main():
    # settings
    #mode = 'MSE'
    mode = 'MLP'
    valid_ratio = 1/6
    np.random.seed(0)


    # hyperparameters
    hidden_size = 1000  # for only MLP
    lr = 0.003
    batch_size = 100
    epochs = 10


    # load data
    train_X, train_y, test_X, test_y = load_data()
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_train = len(train_y)
    n_valid = int(valid_ratio * n_train)
    n_train -= n_valid
    n_test = len(test_y)
    d = train_X.shape[1]
    n_labels = len(labels)
    print(f'Mode: {mode}  #Dim: {d}  #Train: {n_train}  #Valid: {n_valid}  #Test: {n_test}\n')


    # model
    if mode.lower() in ['mse', 'lms', 'widrow-hoff']:
        train_X = add_augment_axis(train_X)
        test_X = add_augment_axis(test_X)

        model = Linear(d+1, n_labels)
    elif mode.lower() in ['mlp', 'nn']:
        model = Model([
            FC(d, hidden_size, sigmoid, deriv_sigmoid),
            FC(hidden_size, n_labels, identity_function, deriv_identity_function),
            ])
    else:
        raise ValueError(f'Unknown mode: {mode}')


    # preprocess and split data
    test_X = normalize(test_X)
    train_X = normalize(train_X)

    valid_X = train_X[n_train:]
    valid_y = train_y[n_train:]
    train_X = train_X[:n_train]
    train_y = train_y[:n_train]


    # train
    model = train(
        model, labels,
        train_X, train_y,
        valid_X, valid_y,
        lr=lr, epochs=epochs, batch_size=batch_size,
        mode=mode,
        )


    # result
    result = test(model, test_X, test_y, labels)
    print_result_in_TeX_tabular_format(result)


if __name__ == '__main__':
    main()
