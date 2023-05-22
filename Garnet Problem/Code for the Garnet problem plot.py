# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:40:37 2023

@author: Qiuhao Wang

project: ICML---draw the Garnet problem graph for both (s,a)- and s-rectangular

"""
import sys
import copy
sys.path.append("..")
import random
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
'''
For s,a-rec case
'''
data_DRPG_s10 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\sa-rec 50samples for DRPG.csv", header=None)
sa_Data_DRPG_s10 = np.array(data_DRPG_s10)
data_RVI_s10 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\sa-rec 50samples for RVI.csv", header=None)
sa_Data_RVI_s10 = np.array(data_RVI_s10)
data_DRPG_s5 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\sa-rec 50samples for DRPG S=5.csv", header=None)
sa_Data_DRPG_s5 = np.array(data_DRPG_s5)
data_RVI_s5 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\sa-rec 50samples for RVI S=5.csv", header=None)
sa_Data_RVI_s5 = np.array(data_RVI_s5)
data_DRPG_s15 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\sa-rec 50samples for DRPG S=15.csv", header=None)
sa_Data_DRPG_s15 = np.array(data_DRPG_s15)
data_RVI_s15 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\sa-rec 50samples for RVI S=15.csv", header=None)
sa_Data_RVI_s15 = np.array(data_RVI_s15)
#%%
'''
For s-rec case
'''
data_DRPG_s10 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\s-rec 50samples for DRPG.csv", header=None)
s_Data_DRPG_s10_old = np.array(data_DRPG_s10)
data_RVI_s10 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\s-rec 50samples for RVI.csv", header=None)
s_Data_RVI_s10_old = np.array(data_RVI_s10)
data_DRPG_s15 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\s-rec 50samples for DRPG S=15.csv", header=None)
s_Data_DRPG_s15_old = np.array(data_DRPG_s15)
data_RVI_s15 = pd.read_csv("F:\Cityu\Research\Robust Optimization\Robust MDP\Policy gradient\AISTATS2023PaperPack\Data and result\s-rec 50samples for RVI S=15.csv", header=None)
s_Data_RVI_s15_old = np.array(data_RVI_s15)
#%%
'''
(s,a)-rec case
'''
gl_time = 200
samp_num = 50

Error_s10 = abs(sa_Data_DRPG_s10 - sa_Data_RVI_s10)
Err_s10_Ave = np.zeros(gl_time)
for i in range(gl_time):
    Err_s10_Ave[i] = np.mean(Error_s10[:,i])
Err_s10_perc_95 = np.zeros(gl_time)
Err_s10_perc_5 = np.zeros(gl_time)
for i in range(gl_time):
    Err_s10_perc_95[i] = np.percentile(Error_s10[:,i],95,interpolation='midpoint')
    Err_s10_perc_5[i] = np.percentile(Error_s10[:,i],5,interpolation='midpoint')

Error_s5 =  abs(sa_Data_RVI_s5 - sa_Data_DRPG_s5)
Err_s5_Ave = np.zeros(gl_time)
for i in range(gl_time):
    Err_s5_Ave[i] = np.mean(Error_s5[:,i])
Err_s5_perc_95 = np.zeros(gl_time)
Err_s5_perc_5 = np.zeros(gl_time)
for i in range(gl_time):
    Err_s5_perc_95[i] = np.percentile(Error_s5[:,i],95,interpolation='midpoint')
    Err_s5_perc_5[i] = np.percentile(Error_s5[:,i],5,interpolation='midpoint')

Error_s15 =  abs(sa_Data_RVI_s15 - sa_Data_DLRPG_s15)
Err_s15_Ave = np.zeros(gl_time)
for i in range(gl_time):
    Err_s15_Ave[i] = np.mean(Error_s15[:,i])
Err_s15_perc_95 = np.zeros(gl_time)
Err_s15_perc_5 = np.zeros(gl_time)
for i in range(gl_time):
    Err_s15_perc_95[i] = np.percentile(Error_s15_1[:,i],95,interpolation='midpoint')
    Err_s15_perc_5[i] = np.percentile(Error_s15_1[:,i],5,interpolation='midpoint')

x = np.arange(0,gl_time,1)
plt.figure(dpi=300)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ay = plt.gca()
y_major_locator = MultipleLocator(5)
ay.yaxis.set_major_locator(y_major_locator)
ay.plot(x, Err_s5_Ave, color='green', label = '$\mathcal{G}$(5,4,3)')
ay.fill_between(x, Err_s5_perc_5, Err_s5_perc_95, color='green', alpha=0.2)
ay.plot(x, Err_s10_Ave, color='blue', label = '$\mathcal{G}$(10,6,3)')
ay.fill_between(x, Err_s10_perc_5, Err_s10_perc_95, color='blue', alpha=0.2)
ay.plot(x, Err_s15_Ave, color='brown', label = '$\mathcal{G}$(15,8,3)')
ay.fill_between(x, Err_s15_perc_5, Err_s15_perc_95, color='brown', alpha=0.2)
plt.xlabel('Number of iterations', fontdict={ 'size'   : 14})
plt.ylabel('Error', fontdict={ 'size'   : 14})
ay.legend(fontsize = 13)
plt.savefig('Error for three Garnet.pdf', bbox_inches='tight')
#%%
'''
s-rec case
'''
gl_time = 200
samp_num = 50

Error_s10 = s_Data_DRPG_s10 - s_Data_RVI_s10
Err_s10_Ave = np.zeros(gl_time)
for i in range(gl_time):
    Err_s10_Ave[i] = np.mean(Error_s10[:,i])
Err_s10_perc_95 = np.zeros(gl_time)
Err_s10_perc_5 = np.zeros(gl_time)
for i in range(gl_time):
    Err_s10_perc_95[i] = np.percentile(Error_s10[:,i],95,interpolation='midpoint')
    Err_s10_perc_5[i] = np.percentile(Error_s10[:,i],5,interpolation='midpoint')

Error_s15 =  s_Data_RVI_s15 - s_Data_DRPG_s15
Err_s15_Ave = np.zeros(gl_time)
for i in range(gl_time):
    Err_s15_Ave[i] = np.mean(Error_s15[:,i])
Err_s15_perc_95 = np.zeros(gl_time)
Err_s15_perc_5 = np.zeros(gl_time)
for i in range(gl_time):
    Err_s15_perc_95[i] = np.percentile(Error_s15_1[:,i],95,interpolation='midpoint')
    Err_s15_perc_5[i] = np.percentile(Error_s15_1[:,i],5,interpolation='midpoint')
x = np.arange(0,gl_time,1)
plt.figure(dpi=300)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ay = plt.gca()
y_major_locator = MultipleLocator(5)
ay.yaxis.set_major_locator(y_major_locator)
ay.plot(x, Err_s15_Ave, color='orange', label = '$\mathcal{G}$(10,5,10)')
ay.fill_between(x, Err_s15_perc_5, Err_s15_perc_95, color='orange', alpha=0.2)
ay.plot(x, Err_s10_Ave, color='olive', label = '$\mathcal{G}$(15,8,15)')
ay.fill_between(x, Err_s10_perc_5, Err_s10_perc_95, color='olive', alpha=0.2)
plt.xlabel('Number of iterations', fontdict={ 'size'   : 14})
plt.ylabel('Relative difference', fontdict={ 'size'   : 14})
ay.legend(fontsize = 13)
plt.savefig('Error for two Garnet.pdf', bbox_inches='tight')