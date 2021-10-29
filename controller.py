'''
only for RRP Hopper
Shihao Feng
2021.10.28
'''

import numpy as np
import pybullet as p
from leg_kinematics import LegKinematicsRRP
import pinocchio as pin

class JointPDController(object):
    def __init__ (self):
        self.kp = np.array([70, 70, 1500])
        self.kd = np.array([2, 2, 10])

    def solve(self, q_d, dq_d, q_state, dq_state):
        q = q_state[7:10]
        dq = dq_state[6:9]
        ddq_d = np.zeros(3) # 期望加速度计算量大，简单地设为0
        tau_d = ddq_d + self.kd*(dq_d - dq) + self.kp*(q_d - q) # (3,)
        return tau_d 

    
class SLIPController(object):
    def __init__(self):
        self.q_d = np.array([0., 0., 0.]) # q_d[2] always 0
        self.dq_d = np.array([0., 0., 0.]) # always 0
        # 关节增益 
        self.kp = np.array([70., 70., 3000.]) 
        self.kd = np.array([2., 2., 10.]) # 阻尼模拟能量损失, 同时防止腿抖动
        # 身体姿态增益
        self.kp_pose = 5. * np.ones(2)  
        self.kd_pose = 1. * np.ones(2)
        # 水平速度增益
        self.kp_vel = 0.1 * np.ones(2) 
        
        self.leg_length_normal = 0.55
        self.RRP = LegKinematicsRRP(L=self.leg_length_normal)

    # private methods
    def __w_to_drpy(self, rpy, w):
        '''
        rpy -> (3,), w -> (3,),drpy -> (3,)
        '''
        H = np.array([[np.cos(rpy[2])/np.cos(rpy[1]), np.sin(rpy[2])/np.cos(rpy[1]), 0.],
                        [-np.sin(rpy[2]), np.cos(rpy[2]), 0.],
                        [np.cos(rpy[2])*np.tan(rpy[1]), np.sin(rpy[2])*np.tan(rpy[1]), 0.]])
        drpy = (H @ w.reshape(-1,1)).ravel()
        return drpy

    def solve(self, q_state, dq_state, robot_state_machine, T_s, vel, dir, F_thrust):
        tau_d = np.zeros(3) # 初始化
        orn_body = q_state[3:7] # 身体姿态 四元数
        rpy = np.array(p.getEulerFromQuaternion(orn_body))
        w_body = dq_state[3:6] # 身体角速度 w
        drpy = self.__w_to_drpy(rpy, w_body)
        q = q_state[7:10] # 关节位置
        dq = dq_state[6:9] # 关节速度

        # 控制虚拟弹簧力
        tau_d[2] = self.kd[2]*(self.dq_d[2] - dq[2]) \
                        + self.kp[2]*(self.q_d[2] - q[2]) 

        # 弹簧伸长时,施加推力抵消能量损耗
        if robot_state_machine == 'THRUST':
            tau_d[2] += F_thrust

        # 触地或者离地时，关节扭矩为0
        if (robot_state_machine == 'LOADING' or robot_state_machine == 'UNLOADING'):
            tau_d[0:2] = np.zeros(2)

        # 弹簧压缩或者伸长时，施加关节扭矩控制身体姿态 
        if (robot_state_machine == 'COMPRESSION' or robot_state_machine == 'THRUST'):     
            # 姿态线性伺服控制
            tau_d[0:2] = - (self.kd_pose*(np.zeros(2) - drpy[0:2]) \
                        + self.kp_pose*(np.zeros(2) - rpy[0:2])) # (2,)          

        # 飞行时，控制足端移动到落地点 
        if robot_state_machine == 'FLIGHT':
            vel_xy_d = np.array([vel*np.cos(dir), vel*np.sin(dir)])
            v_body = dq_state[0:2] # 当前水平速度
            # 相对于H系：坐标系原点与身体坐标系重合，方向与世界坐标系平行
            xy_d = v_body*T_s/2 - self.kp_vel*(vel_xy_d - v_body) # 计算落脚点
            r = q[2] + self.leg_length_normal 
            z_d = - (r**2 - xy_d[0]**2 - xy_d[1]**2)**0.5
            # 转换到B系：身体坐标系
            R_HB = pin.rpy.rpyToMatrix(rpy)
            R_BH = R_HB.T 
            p_H = np.array([xy_d[0], xy_d[1], z_d])
            p_B = (R_BH @ p_H.reshape(-1,1)).ravel() # (3,)
            q_d = self.RRP.IK(p_B)
            self.q_d[0:2] = q_d[0:2]
            # 关节PD控制
            tau_d[0:2] = self.kd[0:2]*(self.dq_d[0:2] - dq[0:2]) \
                        + self.kp[0:2]*(self.q_d[0:2] - q[0:2]) # (2,)
            
        print('tau_d: ', tau_d)
        return tau_d
