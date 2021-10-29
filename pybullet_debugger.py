'''
only for RRP Hopper
Shihao Feng
2021.10.28
'''

import pybullet as p
import numpy as np

class PybulletDebugger(object):
    def __init__(self):
            self.vel_Id = p.addUserDebugParameter('velocity',0.,2.,0.) # m/s
            self.dir_Id = p.addUserDebugParameter('direction',-3.,3.,1.57) # rad
            self.F_thrust_Id = p.addUserDebugParameter('F_thrust',0.,200.,50.) # m
            
    def get_sliders_value(self):
            vel = p.readUserDebugParameter(self.vel_Id)
            dir = p.readUserDebugParameter(self.dir_Id)
            F_thrust = p.readUserDebugParameter(self.F_thrust_Id)
            return vel, dir, F_thrust

    def plot_traj_in_pybullet(self, q_state_pre, q_state, q_toes_state_pre, q_toes_state):
        '''
        Input:  q_toes_state_pre -> 1d array (1,3)
                q_toes_state -> 1d array (1,3)
        '''
        # show body trajectory
        p.addUserDebugLine(q_state_pre[0:3], q_state[0:3], 
                        lineColorRGB=[1,0,0], lineWidth=2, lifeTime=60) 
        # # # show toe trajectory
        # p.addUserDebugLine(q_toes_state_pre[0,:], q_toes_state[0,:], 
        #                 lineColorRGB=[0,1,0], lineWidth=2, lifeTime=10) 
