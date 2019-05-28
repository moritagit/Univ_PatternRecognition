# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from neural_network import *


def load_data(path):
    data = loadmat(path)
    #print(data.keys())
    x = data['x'].T
    y = data['l'].T
    n = data['n'][0, 0]
    d = data['d'][0, 0]
    return x, y, n, d


def compute_loss(y_pred, y_true):
    loss = np.mean((y_pred - y_true)**2)
    return loss


def plot(model, x, l, xx, yy, axy):
    # settings
    plt.clf()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    # scatter
    plt.plot(x[np.where(l==1), 0], x[np.where(l==1), 1], 'bo')
    plt.plot(x[np.where(l==0), 0], x[np.where(l==0), 1], 'bx')

    # contour
    p = model(axy, is_training=False)  # compute classification results
    cs = plt.contour(
        xx, yy, np.reshape(p, xx.shape),
        levels=[-5, 0, 5],
        colors='g',
        )
    plt.clabel(cs)

    # show
    plt.show()
    plt.pause(0.000001)


def train(model, x, y, lr, epochs, fig_path=None,):
    n_sample = len(y)

    xx, yy = np.meshgrid(np.linspace(-1,1), np.linspace(-1,1))
    xf = xx.flatten()[:, np.newaxis]
    yf = yy.flatten()[:, np.newaxis]
    axy = np.concatenate((
        xf,
        yf
        ),axis=1)
    l = (y == 1)
    plt.figure()
    plt.ion()

    for epoch in range(epochs):
        # forward
        y_pred = model(x, is_training=True)
        loss = compute_loss(y, y_pred)

        # backward
        delta = y_pred - y
        model.back_propagate(delta)
        model.update(lr)

        # result
        pred = y_pred
        pred[pred > 0] = 1
        pred[pred < 0] = -1
        n_correct = (pred == y).sum()
        acc = n_correct / n_sample
        print(f'Epoch: {epoch+1}\t#Loss: {loss:.4f}\t#Correct: {n_correct}\tAcc: {acc:.3f}')

        # plot
        #plot(model, x, l, xx, yy, axy)


    # plot
    plot(model, x, l, xx, yy, axy)
    if fig_path:
        plt.savefig(fig_path)
    plt.show()

    return model


def main():
    # settings
    data_type = 'linear'
    #data_type = 'nonlinear'
    #data_type = 'slinear'
    data_path = f'../data/{data_type}-data.mat'
    fig_path = f'../figures/assignment1_1_{data_type}_result.png'
    np.random.seed(1)


    # hyperparameters
    epochs = 500
    lr = 0.9

    # model
    input_size = 2
    output_size = 1
    hidden_size = 10
    model = Model([
        FC(input_size, hidden_size, sigmoid, deriv_sigmoid),
        FC(hidden_size, output_size, identity_function, deriv_identity_function),
        ])


    # load data
    x, y, n, d = load_data(data_path)
    print(f'Data Type: {data_type}  #Sample: {n}  #Dim: {d}\n')


    # train
    model = train(model, x, y, lr, epochs, fig_path=fig_path,)


if __name__ == '__main__':
    main()
