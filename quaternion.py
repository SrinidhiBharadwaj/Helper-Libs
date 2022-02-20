'''
Author: Srinidhi Kalgundi Srinivas
'''
import numpy as np

class Quaternion():

    def __init__(self, real, img):
        self.q_s = real
        self.q_v = np.array([img[0], img[1], img[2]])
        self.q = np.hstack((self.q_s, self.q_v))

    def get_hat_matrix(self, x):
        x = x.reshape(-1, 3)
        x_hat = np.zeros((x.shape[1], x.shape[1]))

        x_hat[1][0] = x[0][2]
        x_hat[2][0] = -x[0][1]

        x_hat[0][1] = -x[0][2]
        x_hat[2][1] = x[0][0]

        x_hat[0][2] = x[0][1]
        x_hat[1][2] = -x[0][0]
        return x_hat

    #Overriding class multiplication
    def __mul__(self, quat):
        real_part = self.q_s * quat.q_s - np.dot(self.q_v.T, quat.q_v)
        q_v_hat = self.get_hat_matrix(self.q_v)
        cross_mul = q_v_hat @ quat.q_v
        img_part = self.q_s*quat.q_v + quat.q_s*self.q_v + cross_mul
        return real_part, img_part

    def get_func_q(self, q_s, q_v, hat_transpose=False):
        q_v_hat = self.get_hat_matrix(q_v)
        if hat_transpose == False:
            img_cols = q_s*np.identity(3) + q_v_hat
        else:
            img_cols = q_s*np.identity(3) - q_v_hat

        q_v = -1*q_v.reshape(-1, 1)
        return np.concatenate((q_v, img_cols), 1)


    def get_rot_matrix(self, verbose=False):
        e_q = self.get_func_q(self.q_s, self.q_v)
        g_q = self.get_func_q(self.q_s, self.q_v, hat_transpose=True)
        if verbose:
            print("E(q): ", e_q)
            print("G(q): ", g_q)
        return e_q @ g_q.T

    def inverse(self, obj):
        if not obj:
            return np.array([self.q_s, -self.q_v])
        else:
            return Quaternion(self.q_s, -self.q_v)

    def __str__(self):
        return f"{[self.q_s, self.q_v]}"

if __name__ == "__main__":

    real = 0.82
    img = [0.06, 0.44, 0.36]
    q = Quaternion(real, img)
    q_inv = q.inverse(False)
    rot_matrix = q.get_rot_matrix(True)
    p = np.array([2, 1, 2])
    print(rot_matrix)


