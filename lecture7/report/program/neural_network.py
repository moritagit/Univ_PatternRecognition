# -*- coding: utf-8 -*-

import numpy as np


def identity_function(x):
    return x


def deriv_identity_function(x):
    return np.ones_like(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class FC(object):
    def __init__(self, input_size, output_size, activate_func, activate_func_deriv,):
        self.W = np.random.rand(input_size, output_size).astype(float)
        self.b = np.zeros(output_size, dtype=float)

        self.dW = None
        self.db = None

        self.x = None
        self.u = None

        self.delta = None

        self.activate_func = activate_func
        self.activate_func_deriv = activate_func_deriv

    def __call__(self, x):
        self.x = x
        self.u = x.dot(self.W) + self.b
        h = self.activate_func(self.u)
        return h

    def back_prop(self, delta, W):
        self.delta = self.activate_func_deriv(self.u) * delta.dot(W.T)
        return self.delta

    def compute_grad(self):
        batch_size = self.delta.shape[0]
        self.dW = self.x.T.dot(self.delta) / batch_size
        self.db = self.delta.mean()


class Model(object):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x, is_training=False):
        if is_training:
            for layer in self.layers:
                x = layer(x)
        else:
            for layer in self.layers:
                x = x.dot(layer.W) + layer.b
                x = layer.activate_func(x)
        return x

    def back_propagate(self, delta):
        W = None
        for i, layer in enumerate(self.layers[::-1]):
            if i == 0:
                layer.delta = delta
            else:
                delta = layer.back_prop(delta, W)
            layer.compute_grad()
            W = layer.W

    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db
        return
