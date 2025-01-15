import os
import numpy as np
import pybullet as p
import pybullet_data
from panda import Panda
from objects import RBOObject
from utils import Trajectory
import time

#np.random.seed(42)

class Env1():

    def __init__(self, visualize=True):
        self.urdfRootPath = pybullet_data.getDataPath()
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)

        # set up camera
        self.set_camera()
        # load some scene objects
        self.plane = p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        self.table_1 = p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, -1, -0.65])
        self.table_2 = p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        self.table_3 = p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, +1, -0.65])

        # call draw AABB to draw boundries
        aabb_tb3 = p.getAABB(self.table_3)
        print(aabb_tb3)
        self.drawAABB(aabb_tb3)

        # load block
        self.block = RBOObject('block')
        self.block.load()
        self.block_position = [0.5, 1, 0.2]
        #self.block_position = [0.5, 0.35, 0.2]
        self.block_quaternion = [0, 0, 0, 1]
        #self.block_quaternion = [0, 0.3827, 0, 0.9239]
        
        self.block.set_position_orientation(self.block_position, self.block_quaternion)
        # load a panda robot
        self.panda = Panda()

    def read_box(self):
        return self.block.get_position(), self.block.get_orientation()


    def reset_box(self):
        self.block.set_position_orientation(self.block_position, self.block_quaternion)


    # input trajectory, output final box position
    def play_traj(self, xi, T, color):
        
        traj = Trajectory(xi, T)
        self.panda.reset_task(xi[0, :], [1, 0, 0, 0])
        self.reset_box()
        sim_time = 0
        
        while sim_time < T:
            self.panda.traj_task(traj, sim_time)
            
            p.stepSimulation()
            line_id = p.addUserDebugLine(
            traj.get_waypoint(sim_time),
            traj.get_waypoint(sim_time+1/240.0),
            color,
            5
        )
            sim_time += 1/240.0 # this is the default step time in pybullet
            time.sleep(1/240.0) # for real-time visualization
        #print(self.read_box())
        
        return self.read_box()
    
    # get feature counts; runs simulation in environment!
    def feature_count(self, xi, color, T=2.0):
        n, _ = np.shape(xi)
        length_reward = 0
        for idx in range(1, n):
            length_reward -= np.linalg.norm(xi[idx, :] - xi[idx-1, :])**2
        #box_position = self.play_traj(xi, T, color)
        box_position, box_orientation = self.play_traj(xi, T, color)
        #box_move_x = abs(box_position[0] - self.block_position[0])
        box_move_y = abs(box_position[1] - self.block_position[1])

        # We want to make sure the pushing does not change the orienation
        # So, the agent will be rewarded as it keeps the orienation fixed
        # To do so, we take the dot product of the desired orientation (self.block_quaternion), with the current one
        aligned_push = np.dot(self.block_quaternion, box_orientation)
        ## Remove the anomolies, for the box_move_y, maybe a good choice would be removing < 0.05
        ## Maybe clip the length rward smaller than -1.5 and larger than
        f = np.array([box_move_y, length_reward, aligned_push])
        return f

    # get reward from feature counts
    def reward(self, f, theta):
        return theta[0] * f[0] + theta[1] * f[1] + theta[2] * f[2]

    def set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-60,
                                     cameraTargetPosition=[0.5, -0.2, 0.0])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)

    def drawAABB(self, aabb):
            
            aabb_min, aabb_max = aabb

            x_min, y_min = aabb_min[0], aabb_min[1]
            x_max, y_max = aabb_max[0], aabb_max[1]
            z_height = aabb_min[2] + 0.05
            # Define the 4 corner points of the boundary in the X-Y plane
            points = [
                [x_min, y_min, z_height],
                [x_max, y_min, z_height],
                [x_max, y_max, z_height],
                [x_min, y_max, z_height]
            ]
            # Draw lines between the corners to create the boundary outline
        
            #p.addUserDebugLine(points[1], points[2], [0, 0, 0], 10)  # Red lines, thickness 2
            p.addUserDebugLine(points[2], points[3], [0, 0, 0], 10)  # Red lines, thickness 2
            #p.addUserDebugLine(points[3], points[0], [0, 0, 0], 10)  # Red lines, thickness 2
