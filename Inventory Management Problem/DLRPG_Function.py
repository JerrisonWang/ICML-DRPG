# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:03:00 2023

@author: Qiuhao Wang

project: ICML---Simple function collection
"""
import random
import numpy as np

'''
Randomize initial distribution
'''
def randrho(n):
    rho = np.zeros(n)
    sum1 = 0
    for i in range(n):
        rho[i] = random.randrange(1,10)
        sum1 += rho[i]
    rho = rho/sum1
    return rho

'''
Randomize tolerance kappa 
'''
def randkappa(n,m):
    kappa = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            kappa[i,j] = round(random.uniform(0.1,0.5), 2)
    return kappa

'''
Randomize tolerance kappa 
'''
def randkappa_s(n):
    kappa = np.zeros(n)
    for i in range(n):
        kappa[i] = round(random.uniform(0.01,1), 2)
    return kappa


'''
Randomize policy
'''
def randpolicy_RMDP(m,n):
    policy = np.zeros((m,n))
    for i in range(m):
        policy_1 = np.zeros(n)
        sum1 = 0
        for j in range(n):
            policy_1[j] = random.randrange(1,10)
            sum1 += policy_1[j]
            pi = policy_1/sum1
        policy[i] = pi
    return policy


