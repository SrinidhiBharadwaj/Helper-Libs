"""
Author: Srinidhi Kalgundi Srinivas
This is a helper script I wrote to assist with solving Homework 2 of ECE276A WI22
"""

import numpy as np

class frame_shift():
    def __init__(self, orientation, camera_pose, intrinsic, baseline):
        self.orientation = orientation
        self.camera_pose = camera_pose
        self.intrinsic = intrinsic
        self.baseline = baseline
        self.R_o_r = np.array([[0., -1., 0],
                        [0., 0., -1.],
                        [1., 0., 0]], dtype=np.float64)
    
    def world_to_cam(self, world_point, verbose="False"):

        #Extrisic calculation
        optical_frame = self.R_o_r @ self.orientation.T #3x3
        if verbose:
            print("Optical frame")
            print(optical_frame)
        
        shifted_pose = world_point - self.camera_pose #1x3
        extrinsic_matrix = optical_frame @ shifted_pose.T #3x3
    
        extrinsic_matrix = np.vstack((extrinsic_matrix, 1)) #4x1
        if verbose:
            print("Extrinsic matrix")
            print(extrinsic_matrix)

        #Projection
        z = extrinsic_matrix[2]
        pi_matrix = (1/z) * extrinsic_matrix
        if verbose:
            print("pi_matrix")
            print(pi_matrix)
        
        #Intrinsic
        #Append a column of zeros
        new_col = np.zeros((3, 1), dtype=np.float64)
        intrinsic = np.hstack((self.intrinsic, new_col))
        if verbose:
            print("Intrinsic matrix")
            print(intrinsic)

        pixel_coord = intrinsic @ pi_matrix
        return pixel_coord

    def cam_to_world(self, left_cam, right_cam, verbose=False):

        intrinsic_distance = left_cam[0][0] - right_cam[0][0]
        #print(intrinsic_distance)
        left_cam = left_cam.T
        left_cam = np.vstack((left_cam, intrinsic_distance))
        left_cam = np.vstack((left_cam, 1)) #Homogeneous transformation

        fsu = self.intrinsic[0][0]
        fsub = fsu*self.baseline
        fsub_row = np.array([[0, 0, 0, fsub]], dtype=np.float64)
        intrinsic = np.hstack((self.intrinsic, np.zeros((3, 1))))
        intrinsic = np.insert(intrinsic, 2, fsub_row, axis=0)

        if verbose:
            print("Intrinsic matrix is: ")
            print(intrinsic)
        
        optical_coord = np.linalg.inv(intrinsic) @ left_cam #4x1 matrix
        z = fsub/intrinsic_distance
        optical_coord = z*optical_coord

        optical_coord = np.delete(optical_coord, 3, axis=0)   
        if verbose:
            print("Optical co-ordinates are: ")
            print(optical_coord)
            
        world_translated_coord = self.R_o_r.T @ optical_coord
        world_translated_coord = self.orientation @ world_translated_coord 
        world_coord = world_translated_coord + self.camera_pose.T

        return world_coord, optical_coord

if __name__ == "__main__":

    a = np.sqrt(3)/2
    b = 1/2
    orientation = np.array([[a, b, 0],
                            [-b, a, 0],
                            [0, 0, 1]], dtype=np.float64)
    pose = np.array([[1, -1, 1]], dtype=np.float64)
    intrinsic = np.array([[10, 0, 100],
                        [0, 10, 100],
                        [0, 0, 1]], dtype=np.float64)

    world_point = np.array([[2.27670901, -2.12200847, 3.5]], dtype=np.float64)
    baseline = 0.5

    frame = frame_shift(orientation, pose, intrinsic, baseline)

    pixel_coord = frame.world_to_cam(world_point, False)
    print("Pixel coordinates are :", pixel_coord)

    z_l = np.array([[102, 85]])
    z_r = np.array([[99, 85]])
    world_coord, optical_co = frame.cam_to_world(z_l, z_r, verbose=True)
    print("World coordinates are: ")
    print(world_coord)
