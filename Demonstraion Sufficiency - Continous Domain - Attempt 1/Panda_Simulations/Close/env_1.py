import os
import numpy as np
import pybullet as p
import pybullet_data
from panda import Panda
from objects import RBOObject
from utils import Trajectory
import time

class Env1():

    def __init__(self, visualize=True):
        # create simulation (GUI)
        self.urdfRootPath = pybullet_data.getDataPath()
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)

        # set up camera
        self.set_camera()

        # load some scene objects
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, -1, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, +1, -0.65])

        # load door
        self.door = RBOObject('door')
        self.door.load()
        self.door_position = [0.75, 0.2, 0.2]
        self.door_quaternion = [1, 1, 1, 1]
        self.door_angle = -0.7
        self.door.set_position_orientation(self.door_position, self.door_quaternion)
        p.resetJointState(self.door.body_id, 1, self.door_angle)

        # load a panda robot
        self.panda = Panda()

    def read_door(self):
        return p.getJointState(self.door.body_id, 1)[0]


    def reset_door(self):
        p.resetJointState(self.door.body_id, 1, self.door_angle)

    # input trajectory, output final box position
    def play_traj(self, xi, T=2.0):
        traj = Trajectory(xi, T)
        self.panda.reset_task(xi[0, :], [1, 0, 0, 0])
        self.reset_door()
        sim_time = 0
        while sim_time < T:
            self.panda.traj_task(traj, sim_time)
            p.stepSimulation()
            sim_time += 1/240.0 # this is the default step time in pybullet
            time.sleep(1/240.0) # for real-time visualization
        return self.read_door()

    # get feature counts; runs simulation in environment!
    def feature_count(self, xi):
        n, m = np.shape(xi)
        height_reward = 0
        for idx in range(1, n):
            height_reward -= abs(xi[idx, -1])
        door_position = self.play_traj(xi)



        f = np.array([door_position, height_reward])
        return f

    # get reward from feature counts
    def reward(self, f, theta):
        return theta[0] * f[0] + theta[1] * f[1]


    def set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=30, cameraPitch=-60,
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