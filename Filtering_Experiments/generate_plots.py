#!/opt/local/bin/python3 --

import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

##############################################################################
##############################################################################
########    3DVAR Experiments with inter-observation time 0.001     ##########
##############################################################################
##############################################################################

### CHECK

'''
sigma = 0.001
gamma = 0.001
tau = 0.001
T = 30

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_001.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.5_predicted_observations_singlescale_001.npz')['prediction'][:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.scatter(np.arange(0,T,tau), v_3DVAR[2,:], label = '3DVAR',s=150, color='darkorange', marker='o')
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)

fig.savefig('./plots/3DVAR_TAU0.001_predicted_observations_singlescale_001.png')

####
sigma = 0.1
gamma = 0.1
tau = 0.001
T = 30

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_001.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.5_predicted_observations_singlescale_001.npz')['prediction'][:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.scatter(np.arange(0,T,tau), v_3DVAR[2,:], label = '3DVAR',s=150, color='darkorange', marker='o')
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)

fig.savefig('./plots/3DVAR_TAU0.001_predicted_observations_singlescale_1.png')

'''

####################################################################
####################################################################
#########       3DVAR with inter-observation time 0.5      #########
####################################################################
####################################################################

sigma = 0.001
gamma = 0.001
tau = 0.5
T = 30

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_001.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.5_predicted_observations_singlescale_001.npz')['prediction'][:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.scatter(np.arange(0,T,tau), v_3DVAR[2,:], label = '3DVAR',s=150, color='darkorange', marker='o')
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)

fig.savefig('./plots/3DVAR_TAU0.5_predicted_observations_singlescale_001.png')

####################################################################
####################################################################
#########       3DVAR with inter-observation time 0.5      #########
####################################################################
####################################################################

sigma = 0.001
gamma = 0.001
tau = 1
T = 30

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_001.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.5_predicted_observations_singlescale_001.npz')['prediction'][:,:int(T/0.5)]
v_3DVAR1 = np.load('./results/3DVAR_TAU1_predicted_observations_singlescale_001.npz')['prediction'][:,:int(T/tau)]

fig, ax = plt.subplots(figsize=(20,8))

ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.scatter(np.arange(0,T,tau), v_3DVAR[2,::2], label = '3DVAR',s=150, color='darkorange', marker='o')
ax.scatter(np.arange(0,T,1), v_3DVAR1[2,:], label = '3DVAR',s=150, color='black', marker='o')
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30, loc='lower left')
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)


fig.savefig('./plots/3DVAR_TAU1_predicted_observations_singlescale_001.png')

##############################################################################
##############################################################################
##################            Noisy 3DVAR Experiment        ##################
##############################################################################
##############################################################################