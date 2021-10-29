#!/usr/local/bin/python
# -*- coding: UTF-8 -*-
'''
Description: Trajectory generator (3D Hopper)
Email: 13247344844@163.com
Author: Shihao Feng
Update time: 2021-10-21
'''

import numpy as np
from scipy.special import comb
import pinocchio as pin

####################################
####################################
######  Bezier traj for leg  #######
####################################
####################################
class BezierTrajGenerator(object):
    def __init__(self):
        pass
    # private method
    def __bernstein_poly(self, k, n, s):
        '''
        Description: The Bernstein polynomial of n, k as a function of t
        Input: k, n -> int
                s -> time scaling [0 ~ 1], float
        Output: Bernstein polynomial
        Note: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
        '''
        B = comb(n, k) * (s ** k) * (1 - s) ** (n - k)
        return B

    def __bezier_traj(self, control_points, T, s):
        '''
        Description: The bezier trajectory as a function of control_points
        Input: control points should be an array:
                [[x1,y1,z1],
                [x2,y2,z2], 
                ..
                [Xn, Yn, Zn]] -> (n,3)
                T_sw -> trajectory period, float
                s -> time scaling [0 ~ 1], float
        Output: p_d, v_d, a_d -> w.r.t hip frame, 1d array (3,)
        '''
        n = control_points.shape[0] - 1 
        B_vec = np.zeros((1, n + 1)) # a vector of each bernstein polynomial
        d_B_vec = np.zeros((1, n)) # B_n-1,i_vec
        dd_B_vec = np.zeros((1,n-1)) # B_n-2,i_vec
        # 位置
        for k in range(n + 1):
            B_vec[0, k] = self.__bernstein_poly(k, n, s) 
        p_d = (B_vec @ control_points).ravel() # (3,) 
        # 速度
        for k in range(n):
            d_B_vec[0, k] = self.__bernstein_poly(k, n-1, s)
        d_control_points = n * (control_points[1:] - control_points[:-1]) 
        v_d = 1/T * (d_B_vec @ d_control_points).ravel()
        # 加速度
        for k in range(n-1):
            dd_B_vec[0, k] = self.__bernstein_poly(k, n-2, s)
        dd_control_points = (n-1) * (d_control_points[1:] - d_control_points[:-1])
        a_d = 1/T/T * (dd_B_vec @ dd_control_points).ravel()
        return p_d, v_d, a_d

    # 参数：轨迹中性点 半步长
    def bezier_sw_traj(self, p_hf0, p_f, h, T_sw, s):
        '''
        Description: The bezier curve for swing phase w.r.t hip frame
        Input:  p_hf0 -> vector from hip frame to foot frame, 1d array (3,) 
                p_f -> foot placement w.r.t foot frame, 1d array (3,)
                h -> ground clearance w.r.t foot frame, float
                T_sw -> trajectory period, float
                s -> time scaling [0 ~ 1], float
        Output: p_d, v_d, a_d -> w.r.t hip frame, 1d array (3,)
        Note: control points should be an array:
                [[x1,y1,z1],
                [x2,y2,z2], 
                ..
                [Xn, Yn, Zn]]
        '''
        # xy_scalar = np.array([-1., -1, -1, -1.1, -1.2, -1.2, 0., 0., 1.2, 1.2, 1.1, 1., 1., 1.]) # (14,)
        # z_scalar = np.zeros(xy_scalar.shape[0]) # default 0, (14,)
        # scalar = np.array([xy_scalar, xy_scalar, z_scalar]).T # (14,3)
        # h_scalar = np.array([0., 0., 0., 0., 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 0., 0., 0., 0.]) # (14,)
        # h_term = h * np.array([np.zeros(h_scalar.shape[0]), np.zeros(h_scalar.shape[0]), h_scalar]).T
        # control_points = scalar * np.array([p_f] * scalar.shape[0]) # (14,3) * (14,3) = (14,3)
        # control_points = control_points + h_term
        # control_points = control_points + np.array([p_hf0] * scalar.shape[0]) # (14,3) + (14,3) = (14,3)

        # like cycloid
        xy_scalar = np.array([-1., -1, -1, -1, 1., 1., 1., 1.]) # (8,)
        z_scalar = np.zeros(xy_scalar.shape[0]) # default 0, (8,)
        scalar = np.array([xy_scalar, xy_scalar, z_scalar]).T # (8,3)
        h_scalar = np.array([0., 0., 0., 2., 2., 0., 0., 0.]) # (8,)
        h_term = h * np.array([np.zeros(h_scalar.shape[0]), np.zeros(h_scalar.shape[0]), h_scalar]).T
        control_points = scalar * np.array([p_f] * scalar.shape[0]) # (8,3) * (8,3) = (8,3)
        control_points = control_points + h_term
        control_points = control_points + np.array([p_hf0] * scalar.shape[0]) # (8,3) + (8,3) = (8,3)

        p_d, v_d, a_d = self.__bezier_traj(control_points, T_sw, s)
        return p_d, v_d, a_d

    def bezier_st_trn_traj(self, p_hf0, p_f, T_st, s): 
        '''
        just for robot translation, without orientation
        '''
        # like cycloid
        xy_scalar = np.array([1., 1., 1., -1., -1., -1.]) # (6,)
        z_scalar = np.zeros(xy_scalar.shape[0]) # default 0, (6,)
        scalar = np.array([xy_scalar, xy_scalar, z_scalar]).T # (6,3)
        control_points = scalar * np.array([p_f] * scalar.shape[0]) # (6,3) * (6,3) = (6,3)
        control_points = control_points + np.array([p_hf0] * scalar.shape[0]) # (6,3) + (6,3) = (6,3)

        p_d, v_d, a_d = self.__bezier_traj(control_points, T_st, s)
        return p_d, v_d, a_d

    # 参数：轨迹起始点 轨迹结束点
    def bezier_sw_traj_2(self, p_hf_start, p_hf_end, h, T_sw, s): 
        '''
        Description: The bezier trajectory for swing phase w.r.t hip frame
        Input:  p_hf_start -> start point of foot w.r.t hip frame, 1d array (3,) 
                p_hf_end -> end point of foot w.r.t hip frame, 1d array (3,) 
                h -> ground clearance w.r.t foot frame, float
                T_sw -> trajectory period, float
                s -> time scaling [0 ~ 1], float
        Output: p_d, v_d, a_d -> w.r.t hip frame, 1d array (3,)
        '''
        p_hf0 = (p_hf_start + p_hf_end) / 2
        p_f = (p_hf_end - p_hf_start) / 2
        p_d, v_d, a_d = self.bezier_sw_traj(p_hf0, p_f, h, T_sw, s)
        return p_d, v_d, a_d

    def bezier_st_trn_traj_2(self, p_hf_start, p_hf_end, T_st, s): 
        p_hf0 = (p_hf_start + p_hf_end) / 2
        p_f = (p_hf_end - p_hf_start) / 2
        p_d, v_d, a_d = self.bezier_st_trn_traj(p_hf0, p_f, T_st, s)
        return p_d, v_d, a_d

