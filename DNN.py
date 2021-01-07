import numpy as np

from plt import *

_INFINITELY_SMALL = 1e-8

# 批版本，每行单独求和
# 注意指针

def softmax(x_):
    x = x_.copy()
    c = np.max(x, axis=1).reshape(-1, 1)
    x -= c
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1).reshape(-1, 1)


def cross_entropy_error(y_, t_):
    y = y_.copy()
    t = t_.copy()
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    return -np.sum(t * np.log(y + _INFINITELY_SMALL)) / y.shape[0]


class Affine(object):
    def __init__(self, W, b, *, dropout=False):
        self.W = W.copy()  # 利用指针连接，会有bug
        self.b = b.copy()
        self.x = None
        self.y = None
        self.dw = None
        self.db = None
        self.dx = None
        self.dy = None
        self.Learn_speed = None
        self.dropout = dropout
        if dropout:
            self.drop1 = None
            self.drop2 = None

    def forward(self, x):
        self.x = x.copy()
        self.y = np.dot(self.x, self.W) + self.b
        if self.dropout:
            self.y[np.expand_dims(~self.drop2, 0).repeat(self.y.shape[0], axis=0)] = 0
        return self.y.copy()

    def backward(self, dy):
        self.dy = dy.copy()
        self.dx = np.dot(self.dy, self.W.T)
        self.dw = np.dot(self.x.T, self.dy)  # / self.dy.shape[0]
        self.db = np.sum(self.dy, axis=0)  # / self.dy.shape[0]
        if self.dropout:
            self.dx[np.expand_dims(~self.drop1, 0).repeat(self.dx.shape[0], axis=0)] = 0
            self.db[~self.drop2] = 0
            self.dw[~self.drop1] = 0
            self.dw[np.expand_dims(~self.drop2, 0).repeat(self.dw.shape[0], axis=0)] = 0
        return self.dx.copy()


class Sigmoid(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.dx = None
        self.dy = None

    def forward(self, x):
        self.x = x.copy()
        self.y = 1 / (1 + np.exp(-self.x))
        return self.y.copy()

    def backward(self, dy):
        self.dy = dy.copy()
        self.dx = (1.0 - self.y) * self.y * self.dy
        return self.dx.copy()


class ReLU(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.dx = None
        self.dy = None
        self.sign = None

    def forward(self, x):
        self.sign = (x <= 0)
        self.x = x.copy()
        self.y = x.copy()
        self.y[self.sign] = 0
        return self.y.copy()

    def backward(self, dy):
        self.dy = dy.copy()
        self.dx = dy.copy()
        self.dx[self.sign] = 0
        return self.dx.copy()


class Softmax_with_loss(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.t = None
        self.dx = None
        self.dy = None
        self.loss = None

    def forward(self, x, t):
        self.x = x.copy()
        self.t = t.copy()
        self.y = softmax(self.x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.y.copy()

    def backward(self, dy=1):
        self.dy = dy  # dy 为 int
        self.dx = (self.y - self.t) / self.y.shape[0]
        return self.dx.copy()


class Neural_Network(object):
    def __init__(self, Affine_layer, activate_layer, out_put_layer, structure, parameter=1, dropout=False):
        self.dropout = dropout
        self.x = None
        self.t = None
        self.structure = structure  # 为tuple，不可变
        self.layer = {}
        self.layer_num = len(self.structure) - 2
        self.loss = []
        self.parameter = {}
        self.dparameter = {}
        if isinstance(parameter, (int, float)):
            self.parameter = self.get_parameter_random(parameter)
        else:
            self.parameter = parameter.copy()

        #生成层
        for i in range(self.layer_num):
            self.layer['Affine' + str(i)] = Affine_layer(self.parameter['W' + str(i)], self.parameter['b' + str(i)],
                                                         dropout=self.dropout)
            if i != self.layer_num - 1:
                self.layer['activate' + str(i)] = activate_layer()
        self.layer['output'] = out_put_layer()


    def get_parameter_random(self, Wide):
        parameter = {}
        for i in range(self.layer_num):
            parameter['W' + str(i)] = np.sqrt(Wide / self.structure[i + 1]) * (
                        np.random.random((self.structure[i + 1], self.structure[i + 2])) - 0.5)
            parameter['b' + str(i)] = np.sqrt(Wide / self.structure[i + 1]) * (
                        np.random.random((self.structure[i + 2],)) - 0.5)
        return parameter

    def refresh_dropout(self, dropout_rate):  # 如不能整除会随机向上/向下取整

        self.parameter['X' + str(0)] = np.repeat(np.array((True,)), int(self.structure[1]))
        self.parameter['X' + str(self.layer_num)] = np.repeat(np.array((True,)), int(self.structure[-1]))
        for i in range(1, self.layer_num):
            self.parameter['X' + str(i)] = np.append(
                np.repeat(np.array((True,)), int(self.structure[i + 1] * (1 - dropout_rate)+0.5)),
                np.repeat(np.array((False,)), int(self.structure[i + 1] * dropout_rate+0.5)))
            np.random.shuffle(self.parameter['X' + str(i)])

        for i in range(self.layer_num):
            self.layer['Affine' + str(i)].drop1 = self.parameter['X' + str(i)]
            self.layer['Affine' + str(i)].drop2 = self.parameter['X' + str(i + 1)]

    def forward(self, x, t, *, save_Y=False):
        self.x = x.copy()
        self.t = t.copy()
        if save_Y:
            Y_layer = []
        for i in range(self.layer_num):
            if i == 0:
                self.layer['Affine' + str(i)].forward(x)
                if save_Y:
                    Y_layer.append((self.layer['Affine' + str(i)].y, 'Affine' + str(i)))
            else:
                self.layer['Affine' + str(i)].forward(self.layer['activate' + str(i - 1)].y)
                if save_Y:
                    Y_layer.append((self.layer['Affine' + str(i)].y, 'Affine' + str(i)))

            if i != self.layer_num - 1:
                self.layer['activate' + str(i)].forward(self.layer['Affine' + str(i)].y)
                if save_Y:
                    Y_layer.append((self.layer['activate' + str(i)].y, 'activate' + str(i)))

        self.layer['output'].forward(self.layer['Affine' + str(self.layer_num - 1)].y, t)
        if save_Y:
            Y_layer.append((self.layer['output'].y, 'output'))
        self.loss.append(self.layer['output'].y)
        # print(self.Layer[7].y)
        if save_Y:
            return Y_layer

    def backward(self, *, save_dx=False):
        self.layer['output'].backward()

        if save_dx:
            dx_layer = []
            dx_layer.append((self.layer['output'].dx, 'output'))

        for i in list(range(self.layer_num))[::-1]:
            if i != self.layer_num - 1:
                self.layer['activate' + str(i)].backward(self.layer['Affine' + str(i + 1)].dx)
                if save_dx:
                    dx_layer.append((self.layer['activate' + str(i)].dx, 'activate' + str(i)))
            if i == self.layer_num - 1:
                self.layer['Affine' + str(i)].backward(self.layer['output'].dx)
                if save_dx:
                    dx_layer.append((self.layer['Affine' + str(i)].dx, 'Affine' + str(i)))
            else:
                self.layer['Affine' + str(i)].backward(self.layer['activate' + str(i)].dx)
                if save_dx:
                    dx_layer.append((self.layer['Affine' + str(i)].dx, 'Affine' + str(i)))

        if save_dx:
            return dx_layer

    def forward_with_weightdecay(self, x, t, decay):
        self.decay = decay
        self.forward(x, t)
        L2 = 0
        for i in range(self.layer_num):
            L2 += np.sum(self.parameter['W' + str(i)] ** 2)
        self.layer['output'].loss = self.loss.pop() + L2 * self.decay / 2
        self.loss.append(self.layer['output'].loss)

    def backward_with_weightdecay(self):
        self.backward()
        for i in range(self.layer_num):
            self.layer['Affine' + str(i)].dw += self.parameter['W' + str(i)] * self.decay

    def updata(self, learn_speed):
        for i in range(self.layer_num):
            self.parameter['W' + str(i)] -= self.layer['Affine' + str(i)].dw * learn_speed
            self.parameter['b' + str(i)] -= self.layer['Affine' + str(i)].db * learn_speed

        self.reload_parameter(self.parameter)

    def updata_numerical(self, learn_speed):
        dparameter = self.get_numerical_derivative()

        for i in range(self.layer_num):
            self.parameter['W' + str(i)] -= dparameter['W' + str(i)] * learn_speed
            self.parameter['b' + str(i)] -= dparameter['b' + str(i)] * learn_speed

        self.reload_parameter(self.parameter)

    def updata_AdaGrad(self, learn_speed, memory_size=None):  # 增加遗忘
        if memory_size:
            for i in range(self.layer_num):
                self.parameter['hW' + str(i)] = np.zeros_like(self.parameter['W' + str(i)])
                self.parameter['pW' + str(i)] = []
                self.parameter['hb' + str(i)] = np.zeros_like(self.parameter['b' + str(i)])
                self.parameter['pb' + str(i)] = []
        for i in range(self.layer_num):

            self.parameter['hW' + str(i)] += self.layer['Affine' + str(i)].dw * self.layer['Affine' + str(i)].dw
            self.parameter['hb' + str(i)] += self.layer['Affine' + str(i)].db * self.layer['Affine' + str(i)].db
            if memory_size:

                self.parameter['pW' + str(i)].append(self.layer['Affine' + str(i)].dw)
                if len(self.parameter['pW' + str(i)]) == memory_size:
                    self.parameter['hW' + str(i)] -= self.parameter['pW' + str(i)].pop(0)

                self.parameter['pb' + str(i)].append(self.layer['Affine' + str(i)].db)
                if len(self.parameter['pb' + str(i)]) == memory_size:
                    self.parameter['hb' + str(i)] -= self.parameter['pW' + str(i)].pop(0)

            self.parameter['W' + str(i)] -= self.layer['Affine' + str(i)].dw * learn_speed * np.sqrt(
                1 / (self.parameter['hW' + str(i)] + _INFINITELY_SMALL))
            self.parameter['b' + str(i)] -= self.layer['Affine' + str(i)].db * learn_speed * np.sqrt(
                1 / (self.parameter['hb' + str(i)] + _INFINITELY_SMALL))

        self.reload_parameter(self.parameter)

    def updata_Momentum(self):

        self.reload_parameter(self.parameter)

    def updata_Adam(self):

        self.reload_parameter(self.parameter)

    def get_parameter(self):
        return self.parameter.copy()

    def get_dparameter(self):
        for i in range(self.layer_num):
            self.dparameter['W' + str(i)] = self.layer['Affine' + str(i)].dw
            self.dparameter['b' + str(i)] = self.layer['Affine' + str(i)].db
        return self.dparameter.copy()

    def reload_parameter(self, parameter):
        self.parameter = parameter
        for i in range(self.layer_num):
            self.layer['Affine' + str(i)].W = self.parameter['W' + str(i)]
            self.layer['Affine' + str(i)].b = self.parameter['b' + str(i)]

    def get_loss(self):
        loss = self.loss
        self.loss = []
        return loss.copy()

    def get_accuracy(self):
        a = np.argmax(self.layer['output'].y, axis=1)
        b = np.argmax(self.layer['output'].t, axis=1)
        return np.sum(a == b) / self.structure[0]

    def get_numerical_derivative(self):  # 使用d(x+dx)-d(x)会得到相当大的误差
        parameter = self.parameter.copy()
        dparameter = self.parameter.copy()
        # loss = self.layer['output'].loss  #loss是float
        for i in range(self.layer_num):

            a = parameter['W' + str(i)].shape
            for m, n in [(m, n) for m in range(a[0]) for n in range(a[1])]:
                x = parameter.copy()
                x['W' + str(i)][m][n] += _INFINITELY_SMALL
                self.reload_parameter(x)
                self.forward(self.x, self.t)
                loss_a = self.layer['output'].loss
                x['W' + str(i)][m][n] -= _INFINITELY_SMALL * 2
                self.reload_parameter(x)
                self.forward(self.x, self.t)
                loss_b = self.layer['output'].loss

                dparameter['W' + str(i)][m][n] = (loss_a - loss_b) / (_INFINITELY_SMALL * 2)

            a = parameter['b' + str(i)].shape[0]
            for n in range(a):
                x = parameter.copy()
                x['b' + str(i)][n] += _INFINITELY_SMALL
                self.reload_parameter(x)
                self.forward(self.x, self.t)
                loss_a = self.layer['output'].loss
                x['b' + str(i)][n] -= _INFINITELY_SMALL * 2
                self.reload_parameter(x)
                self.forward(self.x, self.t)
                loss_b = self.layer['output'].loss
                dparameter['b' + str(i)][n] = (loss_a - loss_b) / (_INFINITELY_SMALL * 2)

        return dparameter.copy()

