# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:19:54 2023

@author: Qiuhao Wang

project: ICML---DRPG solves RMDPs with parameterized transition kernel 
"""
import sys
import copy
import math
sys.path.append("..")
import random
from math import *
import matplotlib.cm as cm
import matplotlib
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from DLRPG_Function import *
import cvxpy as cp
import pandas as pd
import time
import gurobipy
from tqdm import trange
from matplotlib.pyplot import MultipleLocator
np.set_printoptions(threshold=np.inf)

#%%
class GARNET_MDP:
    '''
    Creates a randomized MDP object to pass to an RL algorithm.
    Parameters
    ----------
    state/action number: int
    gamma: double
    transition kernal (P_c): numpy array of shape (S,SA)
    reward (cost): numpy array of shape (SA,S)
    initial distribution (rho): numpy array of shape (1,S)
    branch number and its associated location matrix: numpy array of shape (S,SA)
    '''
    def __init__(self, num_s, num_a):
        self.S_num = num_s
        self.A_num = num_a
        self.gamma = None
        self.P_c = None
        self.b_num = None
        self.rho = None
        self.C_k = None
    
    '''discounted factor and initial distribution'''
    def add_gamma(self, gamma):
        self.gamma = gamma
        
    def add_rho(self, rho):
        self.rho = rho
        
    '''Using Garnet to denote transition and cost'''
    def set_Garnet_sa(self, num_bra, std):
        #nominal distribution transition kernel
        branch_matrix = np.ones((self.S_num,self.S_num*self.A_num))
        P_kernel = np.zeros((self.S_num,self.S_num*self.A_num))
        for i in range(self.S_num*self.A_num):
            sum1 = 0
            p_mid = np.zeros(num_bra)
            for j in range(num_bra):
                p_mid[j] = random.randrange(1,10)
                sum1 += p_mid[j]
            p_mid = p_mid/sum1
            ary_index = list(range(0,self.S_num,1))
            index = random.sample(ary_index,num_bra)
            for k in range(num_bra):
                P_kernel[:,i][index[k]] = p_mid[k]
                branch_matrix[:,i][index[k]] = 0
        '''reward matrix chosen by normal distribution
        cost_ker = np.zeros((self.S_num*self.A_num,self.S_num))
        for i in range(self.S_num*self.A_num):
            for j in range(self.S_num):
                expect_cost = np.random.normal(loc=0, scale=1, size=1)
                cost_ker[i,j] = np.random.normal(loc=expect_cost, scale=std, size=1)
        '''
        '''reward matrix chosed in [0,10] randomly'''
        cost_ker = np.zeros((self.S_num*self.A_num,self.S_num))
        for i in range(self.S_num*self.A_num):
            for j in range(self.S_num):
                cost_ker[i,j] = round(random.gauss(2,2), 2)
        self.P_c = P_kernel
        self.C_k = abs(cost_ker)
        self.b_num = branch_matrix
    
    """
    DRPG for sa-rec RMDP
    """ 
    
    '''
    Occupancy measure
    See in overleaf-ICML eq.3
    '''
    def Garnet_occu_RMDP(self, P_kernel, pi_now):
        #num_s = len(state)
        #num_a = len(action)
        P = np.zeros((self.S_num,self.S_num)) 
        for i in range(self.S_num):
            for j in range(self.A_num):
                P[i] += pi_now[i,j]*P_kernel[:,self.A_num*i+j]
        I = np.identity(self.S_num)
        P_1 = np.linalg.inv(I-self.gamma*P)
        eta = (1-self.gamma) * np.dot(self.rho, P_1)
        #eta = np.dot(rho,P_1)
        return eta
    
    '''
    Compute current value function 
    '''
    
    def Garnet_FixedValue(self, P_kernel, pi_now):
        I = np.identity(self.S_num)
        c = np.zeros(self.S_num)
        for i in range(self.S_num):
            for j in range(self.A_num):
                c[i] += pi_now[i,j]*np.dot(P_kernel[:,self.A_num*i+j], self.C_k[self.A_num*i+j])
        p_pi = np.zeros((self.S_num, self.S_num)) 
        for i in range(self.S_num):
            for j in range(self.A_num):
                p_pi[i] += pi_now[i,j]*P_kernel[:,self.A_num*i+j]
        mid = np.linalg.inv(I-self.gamma*p_pi)
        v = np.dot(mid,c)
        return v
    
    '''
    Compute the partial graident on theta_{i}
    See in overleaf-JLMR eq.13
    '''
    def Garnet_grad_theta(self, pi_now, theta, lamda, eta, V_F, feature_s, feature_sa, empri_P):
        Grad_theta = np.zeros(len(feature_s))
        for index_theta in range(len(feature_s)):
            for i in range(self.S_num):
                for j in range(self.A_num):
                    p_ij = Trans_Psa(theta, lamda, i, j, feature_s, feature_sa, empri_P[:,self.A_num*i+j])
                    d = np.zeros(self.S_num)
                    for k in range(self.S_num):
                        d[k] = feature_s[index_theta, k]/np.dot(lamda, feature_sa[:,self.A_num*i+j])
                    d2 = np.zeros(self.S_num)
                    for k in range(self.S_num):
                        d2[k] = (feature_s[index_theta, k]/np.dot(lamda, feature_sa[:,self.A_num*i+j]) - np.dot(p_ij, d))*(self.C_k[self.A_num*i+j, k] + self.gamma*V_F[k])
                    Grad_theta[index_theta] += (1/(1-self.gamma))*eta[i]*pi_now[i,j]*np.dot(p_ij, d2)
        return Grad_theta
                    
    '''
    Compute the partial gradient on lamda_{i}
    See in overleaf-JLMR eq.13
    '''
    
    def Garnet_grad_lamda(self, pi_now, theta, lamda, eta, V_F, feature_s, feature_sa, empri_P):
        Grad_lamda = np.zeros(len(feature_sa))
        for index_lamda in range(len(feature_sa)):
            for i in range(self.S_num):
                for j in range(self.A_num):
                    p_ij = Trans_Psa(theta, lamda, i, j, feature_s, feature_sa, empri_P[:,self.A_num*i+j])
                    d = np.zeros(self.S_num)
                    for k in range(self.S_num):
                        d[k] = np.dot(theta, feature_s[:,k])*feature_sa[index_lamda, self.A_num*i+j]/((np.dot(lamda, feature_sa[:, self.A_num*i+j]))**2)
                    d2 = np.zeros(self.S_num)
                    for k in range(self.S_num):
                        d2[k] = (np.dot(p_ij, d) - np.dot(theta, feature_s[:,k])*feature_sa[index_lamda, self.A_num*i+j]/((np.dot(lamda, feature_sa[:, self.A_num*i+j]))**2))*(self.C_k[self.A_num*i+j, k] + self.gamma*V_F[k])
                    Grad_lamda[index_lamda] += (1/(1-self.gamma))*eta[i]*pi_now[i,j]*np.dot(p_ij, d2)
        return Grad_lamda
    
    '''
    Solving inner problem
    See in overleaf-JLMR Algorithm 2
    '''
    def InnerPGD_samax_param(self, pi, lamda_c, theta_c, lamda_ini, theta_ini, lamda_step, theta_step, feature_s, feature_sa, lamda_kappa, theta_kappa, P_c):
        lamda_t = lamda_ini
        theta_t = theta_ini
        lamda_T = np.zeros(len(feature_sa))
        theta_T = np.zeros(len(feature_s))
        t = 0
        ones_theta = np.ones(len(feature_s))
        ones_lamda = np.ones(len(feature_sa))
        zeros_lamda = np.zeros(len(feature_sa))
        zeros_theta = np.zeros(len(feature_s))
        V_new = np.zeros(self.S_num)
        while True:
            t += 1
            P_t = Trans_P(theta_t, lamda_t, feature_s, feature_sa, P_c)
            V_now = self.Garnet_FixedValue(P_t, pi)
            #update theta
            eta = self.Garnet_occu_RMDP(P_t, pi)
            Grad_theta = self.Garnet_grad_theta(pi, theta_t, lamda_t, eta, V_now, feature_s, feature_sa, P_c)
            Grad_lamda = self.Garnet_grad_lamda(pi, theta_t, lamda_t, eta, V_now, feature_s, feature_sa, P_c)
            #update theta
            theta = cp.Variable(len(feature_s))
            y = cp.Variable(len(feature_s))
            cons = []
            cons += [ones_theta @ y <= theta_kappa]
            cons += [theta <= theta_c + y]
            cons += [theta >= theta_c - y]
            prob = cp.Problem(cp.Minimize(cp.sum_squares(theta - theta_t) - 2*theta_step*(theta @ Grad_theta)),
                                  cons)
            prob.solve(solver=cp.GUROBI)
            theta_T = theta.value
            #update lamda
            lamda = cp.Variable(len(feature_sa))
            y = cp.Variable(len(feature_sa))
            cons = []
            cons += [lamda >= 1e-5*ones_lamda]
            cons += [ones_theta @ y <= lamda_kappa]
            cons += [lamda <= lamda_c + y]
            cons += [lamda >= lamda_c - y]
            prob = cp.Problem(cp.Minimize(cp.sum_squares(lamda - lamda_t) - 2*lamda_step*(lamda @ Grad_lamda)),
                                  cons)
            prob.solve(solver=cp.GUROBI)
            lamda_T = lamda.value
            gap = np.linalg.norm(V_new - V_now)
            if gap <= 1e-4:
                break
            theta_t = theta_T
            lamda_t = lamda_T
            V_new = V_now
        return theta_t, lamda_t, V_now
    
    '''
    Compute the outer partial derivative
    '''
    
    def Garnet_grad_pi(self, P_kernel, eta, v_pi):
        #num_s = len(state)
        #num_a = len(action)
        Grad = np.zeros((self.S_num,self.A_num))
        for i in range(self.S_num):
            for j in range(self.A_num):
                mid3 = self.C_k[self.A_num*i+j] + self.gamma*v_pi
                q_sa = np.dot(P_kernel[:,self.A_num*i+j],mid3)
                Grad[i,j] = (1/(1-self.gamma))*eta[i]*q_sa
        return Grad

    def Garnet_outer_PGDmin(self, pi_old, step, grad):
        ones_a = np.ones(self.A_num)
        zeros_a = np.zeros(self.A_num)
        pi_new = np.zeros((self.S_num,self.A_num))
        for i in range(self.S_num):
            pi_s = cp.Variable(self.A_num)
            cons = []
            cons += [pi_s @ ones_a == 1]
            cons += [pi_s >= zeros_a]
            prob = cp.Problem(cp.Minimize(pi_s @ (2*step*grad[i]) + cp.sum_squares(pi_s - pi_old[i])),
                                  cons)
            prob.solve(solver=cp.GUROBI)
            pi_new[i] = pi_s.value
        return pi_new
    
    #define the feature function for a specific stata
    def Feature_state(self, state, *args):
        dimension = len(args)
        f_s = np.zeros((dimension, self.S_num))
        for i in range(self.S_num):
            for j in range(dimension):
                up = - (np.linalg.norm(state[:,i] - args[j]))**2
                down = 2*1**2
                sclar = 1/(sqrt(2*(1**2)*math.pi))
                f_s[j,i] = sclar*(up/down)
        return f_s

    #define the feature for a specific state-action
    def Feature_state_action(self, state, action, *args):
        dimension = len(args)
        lenth_state = len(state[1])
        lenth_action = len(action)
        f_sa = np.zeros((dimension, lenth_state*lenth_action))
        for i in range(self.S_num):
            for j in range(self.A_num):
                mid = list(state[:,i])
                mid.append(action[j])
                state_action = np.array(mid)
                for k in range(dimension):
                    up = - (np.linalg.norm(state_action - args[k]))**2
                    down = 2*2**2
                    sclar = 1/(sqrt(2*(2**2)*math.pi))
                    f_sa[k,i*lenth_action+j] = sclar*(up/down) 
        return f_sa
      
#%%
'''
Problem Declaration 
'''
State = np.array([[0.25, 0.5, 0.75, 1, 0.25, 0.5, 0.75, 1], [1.3,-2.1,3.4,-1,2.5,0.5,1.8,-0.8]])
Action = np.array([-3, -1, 5])
num_state = 8
num_action = 3
branch_num = 1
MDP1 = GARNET_MDP(num_state,num_action)
MDP2 = GARNET_MDP(num_state,num_action)
gl_gamma = 0.95
MDP1.add_gamma(gl_gamma)
gl_rho = randrho(MDP1.S_num)
MDP1.add_rho(gl_rho)
MDP1.set_Garnet_sa(num_state, branch_num)
#randomize a empirical transition
MDP2.set_Garnet_sa(num_state, branch_num)
P_empirical = MDP2.P_c
Feature_s = MDP1.Feature_state(State, c_s1, c_s2)
Feature_sa = MDP1.Feature_state_action(State, Action, c_sa1, c_sa2)

'''
Initialization and define the parameters
'''
pi_ini = randpolicy_RMDP(MDP1.S_num, MDP1.A_num)
lamda_ini = np.ones(2)
theta_ini = np.ones(2)
theta_step = 0.01
lamda_step = 0.01
theta_kappa = 1
lamda_kappa = 1
lamda_c = np.array([0.7, 0.6])
theta_c = np.array([0.4, 0.9])
c_s1 = np.array([-1, 2])
c_s2 = np.array([0.3, -0.6])
c_sa1 = np.array([1.3, 2.1, 1])
c_sa2 = np.array([-0.7, 1.5, 0.5])

#%%
'''
Run the DRPG
'''
#DRPG with robustness using different step size
J_DRPG_robust = []
gl_alpha = 0.1
pi = pi_ini
for i in trange(200):
    time.sleep(0.05)
    pi_0 = pi
    theta_t, lamda_t, V_DRPG = MDP1.InnerPGD_samax_param(pi_0, lamda_c, theta_c, lamda_ini, theta_ini, lamda_step, theta_step, Feature_s, Feature_sa, lamda_kappa, theta_kappa, P_empirical)
    prob_ke = Trans_P(theta_t, lamda_t, Feature_s, Feature_sa, P_empirical)
    d = MDP1.Garnet_occu_RMDP(prob_ke, pi_0)
    Grad_pi = MDP1.Garnet_grad_pi(prob_ke, d, V_DRPG)
    pi = MDP1.Garnet_outer_PGDmin(pi_0, gl_alpha, Grad_pi)
    J_DRPG_robust.append(np.dot(gl_rho,V_DRPG)) 
pi_final_robust = pi

J_DRPG_robust_2 = []
gl_alpha = 0.01
pi = pi_ini
for i in trange(200):
    time.sleep(0.05)
    pi_0 = pi
    theta_t, lamda_t, V_DRPG = MDP1.InnerPGD_samax_param(pi_0, lamda_c, theta_c, lamda_ini, theta_ini, lamda_step, theta_step, Feature_s, Feature_sa, lamda_kappa, theta_kappa, P_empirical)
    prob_ke = Trans_P(theta_t, lamda_t, Feature_s, Feature_sa, P_empirical)
    d = MDP1.Garnet_occu_RMDP(prob_ke, pi_0)
    Grad_pi = MDP1.Garnet_grad_pi(prob_ke, d, V_DRPG)
    pi = MDP1.Garnet_outer_PGDmin(pi_0, gl_alpha, Grad_pi)
    J_DRPG_robust_2.append(np.dot(gl_rho,V_DRPG)) 
pi_final_robust_2 = pi

#DRPG without robustness
J_nominal = []
gl_alpha = 0.1
pi = pi_ini
prob_nominal = Trans_P(theta_c, lamda_c, Feature_s, Feature_sa, P_empirical)
for i in trange(200):
    time.sleep(0.05)
    pi_0 = pi
    theta_t, lamda_t, V_DRPG = MDP1.InnerPGD_samax_param(pi_0, lamda_c, theta_c, lamda_ini, theta_ini, lamda_step, theta_step, Feature_s, Feature_sa, lamda_kappa, theta_kappa, P_empirical)
    J_nominal.append(np.dot(gl_rho,V_DRPG))
    V_DRPG = MDP1.Garnet_FixedValue(prob_nominal, pi_0)
    d = MDP1.Garnet_occu_RMDP(prob_nominal, pi_0)
    Grad_pi = MDP1.Garnet_grad_pi(prob_nominal, d, V_DRPG)
    pi = MDP1.Garnet_outer_PGDmin(pi_0, gl_alpha, Grad_pi) 
pi_final_nonrobust = pi

#%%
'''
Plot the graph
'''
x = np.arange(0,200,1)
plt.figure(dpi=300)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ay = plt.gca()
ay.plot(x, J_DRPG_robust, color='blue', linestyle = "--", label = r'DRPG with $\alpha$ = 0.1')
ay.plot(x, J_DRPG_robust_2, color='brown', linestyle = "-.", label = r'DRPG with $\alpha$ = 0.01')
ay.plot(x, J_nominal, color='orange', label = 'non-robust PG')
plt.xlabel('Number of iterations', fontdict={ 'size'   : 14})
plt.ylabel(r'J($\pi$)', fontdict={ 'size'   : 14})
ay.legend(fontsize = 13)
