'''
only for RRP Hopper
Shihao Feng
2021.10.28
'''

import pybullet as p
import numpy as np
from leg_kinematics import LegKinematicsRRP

class GetRobotStates(object):
    def __init__(self, planeId, robotId, jointIds, toeIds):
        self.planeId = planeId
        self.robotId = robotId
        self.jointIds = jointIds
        self.toeIds = toeIds

        self.RRP = LegKinematicsRRP(L=0.55)

        self.num_joint = len(self.jointIds)
        self.q_state = np.zeros(7 + self.num_joint)
        self.dq_state = np.zeros(6 + self.num_joint)
        self.q_toe_state = np.zeros((len(self.toeIds),3))
        self.toe_touching = 0

    def get_body_state(self):
        '''
        output: q_body -> list [7]
                dq_body -> list [6]
        '''
        body_pos_orn = p.getBasePositionAndOrientation(self.robotId)
        q_pos = body_pos_orn[0] # tuple
        q_orn = body_pos_orn[1] # tuple
        body_vel_ang = p.getBaseVelocity(self.robotId)
        dq_linear = body_vel_ang[0] # tuple
        dq_angular = body_vel_ang[1] # tuple
        q_body = np.array(q_pos + q_orn) # list [7]
        dq_body = np.array(dq_linear + dq_angular) # list [6]
        return q_body, dq_body

    def get_joints_state(self):
        '''
        output: q_joint -> 1d array (7,)
                dq_joint -> 1d array (7,) 
        '''
        q_joints = np.zeros(self.num_joint)
        dq_joints = np.zeros(self.num_joint)
        for j in range(self.num_joint):
            # ignore jointReactionForces, appliedJointMotorTorque
            q_joints[j], dq_joints[j], _, _ = p.getJointState(self.robotId, self.jointIds[j]) 
        return q_joints, dq_joints

    def get_toes_state(self):
        '''
        output: q_toes -> nd array (1,3)
        '''
        q_toes = np.zeros((len(self.toeIds),3))
        for j in range(len(self.toeIds)):
            pos = p.getLinkState(self.robotId, self.toeIds[j])[0]
            q_toes[j,:] = pos
        return q_toes
    
    def get_toe_contact_state(self):
        '''
        output: toe_contact_state -> 0 / 1
        '''
        toeId = self.toeIds[0]
        result = p.getContactPoints(self.robotId, self.planeId, toeId, -1)
        if result is (): # 空元组
            toe_contact_state = 0
        else:
            grf = result[0][9]
            grf_dir = result[0][7]
            toe_contact_state = 1
        return toe_contact_state

    def update(self):
        q_body, dq_body = self.get_body_state()
        q_joints, dq_joints = self.get_joints_state()
        self.q_state[0:7] = q_body 
        self.q_state[7:] = q_joints
        self.dq_state[0:6] = dq_body
        self.dq_state[6:] = dq_joints
        
        q_toes = self.get_toes_state()
        self.q_toe_state = q_toes
        toe_contact_state = self.get_toe_contact_state()
        self.toe_touching = toe_contact_state

        return self.q_state, self.dq_state, self.q_toe_state, self.toe_touching
        