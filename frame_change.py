import numpy as np


def get_axis_angle(R):
    angle = np.arccos((np.trace(R) - 1)/2)
    axis = (1/(2*np.sin(angle))) * np.array([[R[2, 1] - R[1, 2]],
                                             [R[0, 2] - R[2, 0]],
                                             [R[1, 0] - R[0, 1]]])
    return axis, angle

def get_quaternion(R):
    axis, angle = get_axis_angle(R)
    quaternion = np.array([[np.cos(angle/2), np.sin(angle/2)*axis]])
    return quaternion

def get_quaternion_inv(q):
    q[0, 1:3] = -q[0, 1:3]
    return q

def rot_world_to_body(R, point):
    return R.T @ point

def rot_body_t_world(R, point):
    return R @ point

def world_to_body(T, point):
    T_inv = np.linalg.inv(T) @ point
    return T_inv

def body_to_world(T, point):
    return T @ point


a = np.cos(np.pi / 3)
b = np.sin(np.pi / 3)
R = np.array([[a, 0, -b],
              [0, 1, 0],
              [b, 0, a]], dtype=np.float64)
p = np.array([[1, 0, 2]], dtype=np.float64)

T = np.hstack((R, p.T))
T = np.vstack((T, np.zeros((1, 4))))
T[3, 3] = 1

axis, angle = get_axis_angle(R)
quaternion = get_quaternion(R)
q_inv = get_quaternion_inv(quaternion)
