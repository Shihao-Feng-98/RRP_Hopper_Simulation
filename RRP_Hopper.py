'''
Description: 3D Hopper
Email: 13247344844@163.com
Author: Shihao Feng
Update time: 2021-10-28
'''

import pybullet as p
import pybullet_data as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from pybullet_debugger import PybulletDebugger
from get_robot_states import GetRobotStates
from leg_kinematics import LegKinematicsRRP
from trajectory_generator import BezierTrajGenerator
from controller import JointPDController, SLIPController

class RobotSimulation(object):
    def __init__(self):
        self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pd.getDataPath()) # build in data
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90,
                                    cameraPitch=-5, cameraTargetPosition=[0, 2, 1.])
        p.setGravity(0,0,-9.81)
        self.sim_frcy = 240
        self.dt = 1./self.sim_frcy
        p.setTimeStep(self.dt) # default time step 240Hz
        self.sim_t = 0.
        # 导入模型
        self.planeId = p.loadURDF("plane.urdf")
        urdfFlags = p.URDF_USE_IMPLICIT_CYLINDER 
        # urdfFlags = p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        self.robotId = p.loadURDF("RRP_Hopper/urdf/RRP_Hopper.urdf", [0., 1., 1.], 
                                np.array(p.getQuaternionFromEuler([0., 0., 0.])), 
                            useFixedBase=False, flags=urdfFlags, globalScaling=1)
                        
        self.mode = '3D' # 'fixed' '1D' '3D'
        # 关节信息
        self.jointIds = self.get_joints_id() 
        self.num_joint = len(self.jointIds) 
        self.motorIds = self.get_motors_id()
        self.num_motor = len(self.motorIds)
        # 身体和关节状态
        self.q_state = np.zeros(7 + self.num_joint) # 关节位置状态
        self.dq_state = np.zeros(6 + self.num_joint) # 关节速度状态
        self.q_state_pre = np.zeros(7 + self.num_joint) 
        self.dq_state_pre = np.zeros(6 + self.num_joint)
        # 足端状态
        self.toeIds = self.get_toes_id()
        self.q_toe_state = np.zeros((len(self.toeIds),3)) # 足端位置状态
        self.toe_touching = 0 # 初始不触地
        self.q_toe_state_pre = np.zeros((len(self.toeIds),3))
        self.toe_touching_pre = 0 
        # 腿长范围 [0.2,0.9]
        self.leg_length_normal = 0.55
        self.T_s = 0.25
        self.stance_start = 0.
        # 有限状态机
        # 'LOADING' 'COMPRESSION' 'THRUST' 'UNLOADING' 'FLIGHT'
        self.robot_state_machine = 'FLIGHT' # 初始状态
        # 记录数据
        self.q_mat = np.zeros((self.sim_frcy * 5, self.num_motor)) # list of current pos
        self.q_d_mat = np.zeros((self.sim_frcy * 5, self.num_motor)) # list of desired pos  
        self.body_height_mat = np.zeros((self.sim_frcy * 5, 1))
        self.rpy_mat = np.zeros((self.sim_frcy * 5, 3))
        self.v_mat = np.zeros((self.sim_frcy * 5, 2))
        self.v_d_mat = np.zeros((self.sim_frcy * 5, 2))
        # 获取类对象
        self.debugger = PybulletDebugger()
        self.get_robot_state = GetRobotStates(self.planeId, self.robotId, self.jointIds, self.toeIds)
        self.traj_generator = BezierTrajGenerator()
        self.controller = SLIPController()
        # 视频记录
        self.logId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "demo_video.mp4")
        # 初始化
        self.init_robot()

###########################
######## 初始化函数 #########
###########################
    def init_robot(self):
        self.set_constraint()
        # 初始化关节位置
        q_init = np.array([0., 0., 0.]) 
        p.resetJointState(self.robotId, self.motorIds[0], q_init[0])
        p.resetJointState(self.robotId, self.motorIds[1], q_init[1])
        p.resetJointState(self.robotId, self.motorIds[2], q_init[2])
        # # 给的基座一个初始速度
        # p.resetBaseVelocity(self.robotId, linearVelocity = [0., 0., 0])
        # 单步运行仿真
        p.stepSimulation()
        # 更新状态
        self.update_robot_state()

###########################
#### pybullet封装函数 ######
###########################
    def get_joints_id(self):
        joints_id = []
        print('All joints infomation:')
        for j in range(p.getNumJoints(self.robotId)):
            info = p.getJointInfo(self.robotId,j)
            print(info)
            joint_type = info[2]
            if (joint_type==p.JOINT_PRISMATIC or joint_type==p.JOINT_REVOLUTE):
                joints_id.append(j)
                # 取消电机闭环控制,执行力矩控制
                p.setJointMotorControl2(self.robotId, j, p.VELOCITY_CONTROL, force=0)
        print('Enable joint ids:', joints_id)
        return joints_id
    
    def get_motors_id(self):
        motors_id = []
        motors_name = ['joint_1', 'joint_2', 'joint_3']
        for j in range(p.getNumJoints(self.robotId)):
            info = p.getJointInfo(self.robotId,j)
            joint_name = str(p.getJointInfo(self.robotId,j)[1].decode())
            if joint_name in motors_name:
                motors_id.append(j)
        print('Motor ids: ', motors_id)
        return motors_id

    def get_toes_id(self):
        toes_id = []
        toes_name = ['link_toe']
        for j in range(p.getNumJoints(self.robotId)):
            link_name = str(p.getJointInfo(self.robotId,j)[12].decode())
            if link_name in toes_name:
                toes_id.append(j)
        print('Toe ids:', toes_id)
        return toes_id

    def set_constraint(self):
        maxForce = 100000
        if self.mode == 'fixed':
            cid = p.createConstraint(self.robotId, -1, -1, -1, 
                                p.JOINT_FIXED, [0,0,0], [0,0,0], [0,1,1])
            p.changeConstraint(cid, maxForce = maxForce)
        if self.mode == '1D':
            cid = p.createConstraint(self.robotId, -1, -1, -1, 
                                p.JOINT_PRISMATIC, [0,0,1], [0,0,0], [0,1,1])
            p.changeConstraint(cid, maxForce = maxForce)
        if self.mode == '3D':
            pass

    def set_motor(self, q_d):
        max_force = 10 * np.ones(self.num_motor) # Nm
        mode = p.POSITION_CONTROL
        p.setJointMotorControlArray(self.robotId, self.motorIds, mode, q_d, forces=max_force)
        
    def set_motor_tau(self, tau_d):
        mode = p.TORQUE_CONTROL
        for j in range(self.num_motor):
            p.setJointMotorControl2(self.robotId, self.motorIds[j], mode, force=tau_d[j])

###########################
######### 控制函数 #########
###########################
    def update_state_machine(self):
        r_thres_hold = 0.95 # 阈值
        leg_length = self.q_state[9] + self.leg_length_normal
        d_leg_length = self.dq_state[8]
        if self.robot_state_machine == 'LOADING': # 触地
            # 虚拟弹簧压缩超过阈值，切换状态
            if leg_length < self.leg_length_normal * r_thres_hold:
                self.robot_state_machine = 'COMPRESSION'

        elif self.robot_state_machine == 'COMPRESSION': # 压缩
            # 压缩到最短，腿长开始增加，切换状态
            if d_leg_length > 0:
                self.robot_state_machine = 'THRUST'

        elif self.robot_state_machine == 'THRUST': # 伸长
            # 虚拟弹簧伸长超过阈值，切换状态
            if leg_length > self.leg_length_normal * r_thres_hold:
                self.robot_state_machine = 'UNLOADING'

        elif self.robot_state_machine == 'UNLOADING': # 离地
            # 足端离地，切换状态
            if self.toe_touching == 0:
                self.robot_state_machine = 'FLIGHT'

        elif self.robot_state_machine == 'FLIGHT': # 腾空
            # 足端触地，切换状态
            if self.toe_touching == 1:
                self.robot_state_machine = 'LOADING'
        print('robot state: ', self.robot_state_machine)

    def update_last_Ts(self):
        if (self.toe_touching_pre == 0 and self.toe_touching == 1):
            # 进入支撑相
            self.stance_start = self.sim_t
        if (self.toe_touching_pre == 1 and self.toe_touching == 0):
            # 进入摆动相
            stance_end = self.sim_t
            self.T_s = stance_end - self.stance_start

    def update_robot_state(self):
        # 更新状态
        state = self.get_robot_state.update()
        self.q_state = state[0] 
        self.dq_state = state[1]
        self.q_toe_state = state[2] 
        self.toe_touching = state[3] # 足端触底状态
        self.update_last_Ts() # 支撑相持续时间
        self.update_state_machine() # 更新状态机
        self.sim_t += self.dt # 更新仿真时间
        print('v_body: ', self.dq_state[0:2])
        print('simulation time: ', self.sim_t)
        # 仿真内身体和足端轨迹显示,会减慢仿真速度
        self.debugger.plot_traj_in_pybullet(self.q_state_pre, self.q_state, 
                                self.q_toe_state_pre, self.q_toe_state) 
        # 更新上一状态
        self.q_state_pre = self.q_state.copy() # 防止引用 
        self.dq_state_pre = self.dq_state.copy()
        self.q_toe_state_pre = self.q_toe_state.copy()
        self.toe_touching_pre = self.toe_touching

    def traj_generation(self, pxy_start, pxy_end):
        '''
        t should belong to [0,T_tt]
        '''
        t = self.sim_t % 1
        T_sw = 0.5
        T_st = 0.5
        h = 0.2
        p_hf_start = np.array([pxy_start[0], pxy_start[1], -0.6])
        p_hf_end = np.array([pxy_end[0], pxy_end[1], -0.6])
        if t <= T_sw:
            s = t / T_sw
            p_d, v_d, a_d = self.traj_generator.bezier_sw_traj_2(p_hf_start, p_hf_end, h, T_sw, s)
        elif t > T_sw:
            s = (t - T_sw) / T_st
            p_d, v_d, a_d = self.traj_generator.bezier_st_trn_traj_2(p_hf_start, p_hf_end, T_st, s)
        else:
            print('Error: t out of range')
        return p_d, v_d, a_d

    def robot_position_controller(self, pxy_start, pxy_end):
        # 调试用，需固定基座
        self.RRP = LegKinematicsRRP(L = self.leg_length_normal)
        p_d, v_d, a_d = self.traj_generation(pxy_start, pxy_end)
        q_d = self.RRP.IK(p_d)
        # q = self.q_state[7:10]
        # J_a, _ = self.RRP.Jacobian(q) # 当前雅克比
        # dq_d = (np.linalg.inv(J_a) @ v_d.reshape(-1,1)).ravel()
        # q_d = np.array([0.5, 0.5, 0.1]) # step response
        dq_d = np.zeros(3)
        controller = JointPDController()
        tau_d = controller.solve(q_d, dq_d, self.q_state, self.dq_state)
        # 更新绘图数据
        self.q_d_mat[:-1] = self.q_d_mat[1:]
        self.q_d_mat[-1] = q_d
        return tau_d

    def robot_SLIP_controller(self, vel, dir, F_thrust):
        # SLIP controller
        tau_d = self.controller.solve(self.q_state, self.dq_state, 
                                        self.robot_state_machine, 
                                    self.T_s, vel, dir, F_thrust)
        # 更新绘图数据
        self.q_d_mat[:-1] = self.q_d_mat[1:]
        self.q_d_mat[-1] = np.array([0,0,0])
        return tau_d


###########################
######### 绘图函数 #########
###########################
    def init_plot(self):
        self.fig = plt.figure(figsize=(6, 6))
        motors_name = ['joint_1','joint_2','joint_3']
        self.q_d_lines = [] 
        self.q_lines = [] 
        for i in range(self.num_motor):
            plt.subplot(self.num_motor, 1, i + 1)
            q_d_line, = plt.plot(self.q_d_mat[:, i], '-')
            q_line, = plt.plot(self.q_mat[:, i], '--')
            self.q_d_lines.append(q_d_line)
            self.q_lines.append(q_line)
            plt.ylabel('{}'.format(motors_name[i]))
            plt.ylim([-2, 2])
        plt.xlabel('Simulation steps')
        self.fig.tight_layout()
        plt.draw()

    def update_plot(self):
        for i in range(self.num_motor):
            self.q_d_lines[i].set_ydata(self.q_d_mat[:, i])
            self.q_lines[i].set_ydata(self.q_mat[:, i])
        plt.draw()
        plt.pause(1/1000.)

    def horizon_plot(self):
        fig = plt.figure(figsize=(6, 9))
        if self.mode == 'fixed':
            motors_name = ['joint_1','joint_2','joint_3']
            ax1 = fig.add_subplot(3,1,1)
            ax1.plot(self.q_d_mat[:, 0], '-')
            ax1.plot(self.q_mat[:, 0], '--')
            ax1.set_ylabel('{}'.format(motors_name[0]))
            ax1.set_ylim([-0.785, 0.785])
            ax2 = fig.add_subplot(3,1,2)
            ax2.plot(self.q_d_mat[:, 1], '-')
            ax2.plot(self.q_mat[:, 1], '--')
            ax2.set_ylabel('{}'.format(motors_name[1]))
            ax2.set_ylim([-0.785, 0.785])
            ax3 = fig.add_subplot(3,1,3)
            ax3.plot(self.q_d_mat[:, 2], '-')
            ax3.plot(self.q_mat[:, 2], '--')
            ax3.set_ylabel('{}'.format(motors_name[2]))
            ax3.set_ylim([-0.5, 0.5])
        else:
            name = ['roll','pitch','vel_x','vel_y','body_height']
            rpy_d_mat = np.zeros((self.sim_frcy * 5, 3))
            ax1 = fig.add_subplot(5,1,1)
            ax1.plot(rpy_d_mat[:, 0], '-')
            ax1.plot(self.rpy_mat[:, 0], '--')
            ax1.set_ylabel('{}'.format(name[0]))
            ax1.set_ylim([-0.5, 0.5])
            ax2 = fig.add_subplot(5,1,2)
            ax2.plot(rpy_d_mat[:, 1], '-')
            ax2.plot(self.rpy_mat[:, 1], '--')
            ax2.set_ylabel('{}'.format(name[1]))
            ax2.set_ylim([-0.5, 0.5])
            ax4 = fig.add_subplot(5,1,3)
            ax4.plot(self.v_d_mat[:, 0], '-')
            ax4.plot(self.v_mat[:, 0], '--')
            ax4.set_ylabel('{}'.format(name[2]))
            ax4.set_ylim([-3, 3])
            ax5 = fig.add_subplot(5,1,4)
            ax5.plot(self.v_d_mat[:, 1], '-')
            ax5.plot(self.v_mat[:, 1], '--')
            ax5.set_ylabel('{}'.format(name[3]))
            ax5.set_ylim([-3, 3])
            ax6 = fig.add_subplot(5,1,5)
            ax6.plot(self.body_height_mat[:, 0], '--')
            ax6.set_ylabel('{}'.format(name[4]))
            ax6.set_ylim([0, 1.5])

        plt.xlabel('Simulation steps')
        plt.show()

###########################
####### 循环运行函数 ########
###########################
    def run(self):
        # disable real-time simulation
        p.setRealTimeSimulation(0) 
        for i in range (int(100*self.sim_frcy)):    
            vel_d, dir_d, F_thrust = self.debugger.get_sliders_value()

            # 执行控制
            if self.mode == 'fixed':
                tau_d = self.robot_position_controller(np.array([-0.2,-0.2]), np.array([0.2,0.2]))
            else:
                tau_d = self.robot_SLIP_controller(vel_d, dir_d, F_thrust)
            self.set_motor_tau(tau_d)

            p.stepSimulation()
            # 更新相机视角
            p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90,
                            cameraPitch=-5, cameraTargetPosition=[self.q_state[0],self.q_state[1],1])

            # 更新绘图数据
            self.q_mat[:-1] = self.q_mat[1:]
            self.q_mat[-1] = self.q_state[7:10]
            self.body_height_mat[:-1] = self.body_height_mat[1:]
            self.body_height_mat[-1] = self.q_state[2]
            rpy = np.array(p.getEulerFromQuaternion(self.q_state[3:7]))
            self.rpy_mat[:-1] = self.rpy_mat[1:]
            self.rpy_mat[-1] = rpy
            self.v_d_mat[:-1] = self.v_d_mat[1:]
            self.v_d_mat[-1] = np.array([vel_d*np.cos(dir_d), vel_d*np.sin(dir_d)])
            self.v_mat[:-1] = self.v_mat[1:]
            self.v_mat[-1] = self.dq_state[0:2]

            # 更新状态
            self.update_robot_state()

            # # 实时更新绘图
            # if int(self.sim_t / self.dt) % 24 == 0:
            #         self.update_plot()

            # # 一次性绘图 
            # if int(self.sim_t/self.dt) ==30*self.sim_frcy:
            #     self.horizon_plot()

            time.sleep(1/1000.) # would not affect the simulation
        p.stopStateLogging(self.logId)
        p.disconnect()


if __name__ == '__main__':
    a1_sim = RobotSimulation()
    a1_sim.run()