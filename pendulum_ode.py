'''
Written to understand how ODEs work
Note: For my understanding only

Author: Srinidhi Kalgundi Srinivas
'''

import numpy as np
from matplotlib import pyplot as plt

class pendulum():
    def __init__(self):
        self.g = 9.8 #Accelaration due to gravity
        self.mu = 0.4
        self.L = 2
    
    def calc_double_diff_theta(self, theta_t, theta_dot):
        theta_dd = -(self.g/self.L) * np.sin(theta_t) - self.mu * theta_dot
        return theta_dd

if __name__ == "__main__":

    theta_0 = np.pi/2
    theta_dot_0 = 10
    pend = pendulum()

    tau = 0.1
    t = np.linspace(0, tau, 1000)
    theta_t = theta_0
    theta_dot = theta_dot_0

    theta_all, theta_dot_all = [], []

    for i in t:
        theta_dd = pend.calc_double_diff_theta(theta_t, theta_dot)
        theta_t += theta_dot * tau
        theta_dot += theta_dd * tau
        theta_all.append(theta_t)
        theta_dot_all.append(theta_dot)

    plt.plot(theta_all, theta_dot_all)
    plt.show()

    
