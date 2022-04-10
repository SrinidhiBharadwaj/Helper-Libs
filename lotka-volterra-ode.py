'''
Written to understand how ODEs work
Note: For my understanding only

Author: Srinidhi Kalgundi Srinivas
'''
import numpy as np
from matplotlib import pyplot as plt


class LV_Model():
    def __init__(self, alpha, beta, gamma, delta, x_0, y_0):
        self.a = alpha
        self.b = beta
        self.c = gamma
        self.d = delta
        self.prey_init = x_0
        self.pred_init = y_0

    def calc_prey_rate_change(self, x_t, y_t):
        x_dot_t = self.a * x_t - (self.b * x_t * y_t)
        return x_dot_t

    def calc_pred_rate_change(self, x_t, y_t):
        y_dot_t = (self.c * x_t * y_t) - self.d * y_t
        return y_dot_t

if __name__ == "__main__":
    alpha, beta, gamma, delta = 0.7, 0.5, 0.3, 0.2
    x_0, y_0 = 1, 0.5

    t = np.linspace(0, 15, 100)
    x_dot_t, y_dot_t = [], []

    lv = LV_Model(alpha, beta, gamma, delta, x_0, y_0)
    for i in t:
        x_0 += lv.calc_prey_rate_change(x_0, y_0)
        y_0 += lv.calc_pred_rate_change(x_0, y_0)
        x_dot_t.append(x_0)
        y_dot_t.append(y_0)

    p = plt.plot(t, x_dot_t, 'r', t, y_dot_t, 'g', linewidth = 2)
    plt.show()