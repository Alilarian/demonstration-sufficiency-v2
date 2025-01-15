import numpy as np
import pybullet as p
import pybullet_data
import os


class Panda():

    def __init__(self, basePosition=[0,0,0]):
        self.urdfRootPath = pybullet_data.getDataPath()
        self.panda = p.loadURDF(os.path.join(self.urdfRootPath,"franka_panda/panda.urdf"),
                useFixedBase=True, basePosition=basePosition)

    def reset(self):
        init_pos = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.05, 0.05]
        for idx in range(len(init_pos)):
            p.resetJointState(self.panda, idx, init_pos[idx])     

    def reset_task(self, ee_position, ee_quaternion):
        self.reset()
        init_pos = self.inverse_kinematics(ee_position, ee_quaternion)
        for idx in range(len(init_pos)):
            p.resetJointState(self.panda, idx, init_pos[idx])

    def reset_joint(self, joint_position):
        init_pos = list(joint_position) + [0.0, 0.0, 0.05, 0.05]
        for idx in range(len(init_pos)):
            p.resetJointState(self.panda, idx, init_pos[idx])

    def read_state(self):
        joint_position = [0]*9
        joint_states = p.getJointStates(self.panda, range(9))
        for idx in range(9):
            joint_position[idx] = joint_states[idx][0]
        ee_states = p.getLinkState(self.panda, 11)
        ee_position = list(ee_states[4])
        ee_quaternion = list(ee_states[5])
        gripper_contact = p.getContactPoints(bodyA=self.panda, linkIndexA=10)
        state = {}
        state['joint_position'] = np.array(joint_position)
        state['ee_position'] = np.array(ee_position)
        state['ee_quaternion'] = np.array(ee_quaternion)
        state['gripper_contact'] = len(gripper_contact) > 0
        return state

    def inverse_kinematics(self, ee_position, ee_quaternion):
        #print("ee_position: ", ee_position)
        return p.calculateInverseKinematics(self.panda, 11, list(ee_position), list(ee_quaternion), maxNumIterations=5)

    def traj_task(self, traj, time):
        state = self.read_state()
        pd = traj.get_waypoint(time)
        qd = self.inverse_kinematics(pd, [1, 0, 0, 0])
        q_dot = 100 * (qd - state["joint_position"])
        p.setJointMotorControlArray(self.panda, range(9), p.VELOCITY_CONTROL, targetVelocities=list(q_dot))
        #p.setJointMotorControlArray(self.panda, range(9), p.POSITION_CONTROL, targetPositions=list(qd))


    def traj_joint(self, traj, time):
        state = self.read_state()
        qd = traj.get_waypoint(time)
        q_dot = 100 * (qd - state["joint_position"])
        p.setJointMotorControlArray(self.panda, range(9), p.VELOCITY_CONTROL, targetVelocities=list(q_dot))