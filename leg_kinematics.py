#!/usr/local/bin/python
# -*- coding: UTF-8 -*-

'''
Description: Kinematics of RRP leg 
Email: 13247344844@163.com
Author: Shihao Feng
Update time: 2021-10-24
'''

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy import integrate
import cmath

class LegKinematics(object):
    def __init__(self):
        pass

    def PoE(self, screw, q, M):
        '''
        Description: Product of exponential formula
        Input: screw -> [Motion1 Motion2 ... Motionn], list 
                q -> 1d array (n,)
                M -> home configuration, nd array (4,4)
        OutPut: T_hf -> homogeneous matrix from hip to foot 
                        ,nd array (4,4)
        '''
        T_hf = M 
        a = range(len(q))
        for i in a[::-1]: # (a[::-1] -> i:j:step) (i:j -> :) (-1 -> step)
            temp = T_hf.copy()
            T_hf = pin.exp6(q[i] * screw[i]).homogeneous @ temp
        return T_hf

    def Jacobian_g(self, screw, q):
        '''
        Description: Geometric Jacobian w.r.t spatial frame
        Input: screw -> [Motion1 Motion2 ... Motionn], list 
                q -> 1d array (n,)
        OutPut: J_g -> geometric Jacobian, nd array (6,n)
        Note: Motion -> v,w
                V -> v,w
        '''
        n = len(q)
        J_g = screw[0].vector.reshape((-1,1)) # (6,1)
        M = np.eye(4)
        for i in range(1,n):
            q_pie = q.copy()
            q_pie[i:] = np.zeros(n-i) # (3,)
            T = self.PoE(screw, q_pie, M) # (4,4)
            T_motion = pin.log6(T) # Motion
            T_SE3 = pin.exp6(T_motion) # SE3
            X = T_SE3.action # adjoint (6,6)
            J_g = np.hstack((J_g, X @ screw[i].vector.reshape((-1,1))))
        return J_g

    def Jacobian_a(self, screw, q, p):
        '''
        Description: Analytical Jacobian w.r.t spatial frame
        Input: screw -> [Motion1 Motion2 ... Motionn], list 
                q -> 1d array (n,)
                P -> general position [XYZ, RPY], 1d array (6,)
                    or [XYZ], 1d array(3,)
        OutPut: J_a -> analytical Jacobian, nd array (6,n) or (3,n)
        Note: Motion -> v,w
                x_dot -> (XYZ_dot, RPY_dot)
        '''
        if len(p) == 6: # for manipulator
            B = np.array([[np.cos(p[4])*np.cos(p[5]), -np.sin(p[5]),0],
                            [np.cos(p[4])*np.sin(p[5]), np.cos(p[5]),0],
                            [0,0,1]]) # w = B @ RPY_dot (3,3)
            B_inv = np.mat(B).I.A # (3,3)
            temp = np.hstack((np.eye(3), -pin.skew(p[0:3]))) # (3,6)
            temp2 = np.hstack((np.zeros(3,3), B_inv)) # (3,6)
            E = np.vstack((temp, temp2)) # (6,6)
        elif len(p) == 3: # for leg
            E = np.hstack((np.eye(3), -pin.skew(p[0:3]))) # (3,6)
        else:
            print('P should be (3,) or (6,)')
        J_g = self.Jacobian_g(screw, q) # (6,n)
        J_a = E @ J_g #(6,n) or (3,n)
        return J_a


####################################
####################################
###############  RRP  ##############
####################################
####################################
class LegKinematicsRRP(LegKinematics):
    def __init__(self, L=0.):
        '''
        Input: L : kinematics parameters, normal length
        Note: the hip frames and body frame  
                  z ^ 
                    |
                    |
            <-------/
            x      /
                  / y
        '''
        self.L = L

    def FK(self, q):
        '''
        Description: Forward kinematics of prismatic leg w.r.t spatial frame
        Input: q -> 1d array (3,)
        Output: T_hf -> homogeneous matrix from hip to foot 
                        ,nd array (4,4)
        Note: the rotation of each frame of the leg is the same as the body frame
        '''
        M = np.array([[1.,0.,0.,0.], 
                        [0.,1.,0.,0.], 
                        [0.,0.,1.,-self.L], 
                        [0.,0.,0.,1.]]) # home matrix
        w1 = np.array([1, 0, 0]) # (3,)
        v1 = np.array([0, 0, 0]) # (3,)
        S1 = pin.Motion(v1,w1) # Motion
        w2 = np.array([0, 1, 0]) # (3,)
        v2 = np.array([0, 0, 0]) # (3,)
        S2= pin.Motion(v2,w2) # Motion
        w3 = np.array([0, 0, 0]) # (3,)
        v3 = np.array([0, 0, -1]) # (3,)
        S3 = pin.Motion(v3,w3) # Motion
        screw = [S1, S2, S3]
        T_hf = self.PoE(screw, q, M)
        return T_hf

    def IK(self, p):
        '''
        Description: Inverse kinematics of prismatic leg
        Input: p -> 1d array (3,)
        Output: q -> 1d array (3,) 
        '''
        r = np.linalg.norm(p) # leg length
        hip_roll = np.arctan(- p[1] / p[2]) # q1 x
        hip_pitch = np.arcsin(- p[0] / r) # q2 y
        q = np.array([hip_roll, hip_pitch, r - self.L])
        return q

    def Jacobian(self, q):
        '''
        Description: Analytical Jacobian and geometric Jacobian
                    of prismatic leg w.r.t spatial frame
        Input: q -> 1d array (3,)
        Output: J_a -> nd array (3,n)
                J_g -> nd array (6,n)
        '''
        w1 = np.array([1, 0, 0]) # (3,)
        v1 = np.array([0, 0, 0]) # (3,)
        S1 = pin.Motion(v1,w1) # Motion
        w2 = np.array([0, 1, 0]) # (3,)
        v2 = np.array([0, 0, 0]) # (3,)
        S2= pin.Motion(v2,w2) # Motion
        w3 = np.array([0, 0, 0]) # (3,)
        v3 = np.array([0, 0, -1]) # (3,)
        S3 = pin.Motion(v3,w3) # Motion
        screw = [S1, S2, S3]
        p = self.FK_RRP(q)[0:3,3]
        J_a = self.Jacobian_a(screw,q,p)
        J_g = self.Jacobian_g(screw,q)
        return J_a, J_g

